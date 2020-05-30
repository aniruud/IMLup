# This file defines the logprob function which is used in the MCMC_parallelized file for the MCMC run. It calculates the mm continuum and scattered light image for the disk given disk parameters and observations in the mm continuum. The function returns the logprob value by looking at the difference of the observation and calculation of the observed disk. The two observations are weighted equally so that fitting one particular observation is not favoured. The disk setup and scattered light image are similar to those from the Disk_setup and Vertical_tests_plots notebooks

import numpy as np

import disklab
from disklab.natconst import au
from disklab import radmc3d
from disklab.natconst import MS, LS, Mea, AU

from astropy.io import fits
from radmc3dPy import *
from imgcube import imagecube

import dsharp_opac as opacity
import dsharp_helper as dh

import astropy.units as u
import astropy.constants as c  

import os
import glob
import tempfile
from powerlaw_distri import get_powerlaw_dust_distribution


def logprob(parameters, observed_intensity, observed_intensity_error, observed_radius):
    
    # returning an extremely low logprob in case of unrealistic physical conditions
    if parameters[0]<=0 or parameters[5]<=0 or parameters[5]>0.5:
        return -1e6
    
    
    try:
        # The different indices in the parameters list correspond to different physical paramters
        sigma_coeff = parameters[0]
        sigma_exp   = parameters[1]
        size_exp    = parameters[2]
        amax_coeff  = parameters[3]
        amax_exp    = parameters[4]        
        d2g_coeff   = parameters[5]
        d2g_exp     = parameters[6]

        #All files are outputed to a temporary python directory
        temp_directory = tempfile.TemporaryDirectory()
        temp_path = temp_directory.name
        # os.chdir(temp_directory.name)

        # opac_name = 'DSHARP'  # choices: DSHARP, Draine

        # define wave length and particle size grids
        lam_opac = np.logspace(-5, 1, 200)
        a_opac     = np.logspace(-5, -1, 10)

        # define optical constants and material density
        opac_fname = 'dustkappa_default'
        diel_const, rho_s = opacity.get_dsharp_mix()

        # call the Mie calculation & store the opacity in a npz file
        opac_dict = opacity.get_opacities(
            a_opac, lam_opac, rho_s=rho_s, diel_const=diel_const, extrapolate_large_grains=True)
        opacity.write_disklab_opacity(opac_fname, opac_dict, path=temp_path)

        # Mass, t and luminosity from Avenhaus 2018
        mstar = 0.7 * MS
        lstar = 1.56 * LS
        tstar = 4266.00
        mdust = 121 * Mea * 5
        mdisk = mdust / 0.1
        # flang = 0.05  # ??
        zrmax = 1.0  # ??
        nr = 100
        rin = 0.1 * au
        r_c = 100 * au  # ??
        rout = 400 * au  # 400au from avenhaus paper  #DSHARP Huang 2018 says 290 au
        alpha = 1e-3

        opac_params = ['dustcomponents', {'method': 'simplemixing'}]

        d = disklab.DiskRadialModel(mstar=mstar, lstar=lstar, tstar=tstar,
                                    mdisk=mdisk, nr=nr, alpha=alpha, rin=rin, rout=rout)

        d.make_disk_from_simplified_lbp(sigma_coeff, 3 * r_c, sigma_exp)
        
        if d.mass/mstar > 0.2:
        
            return -1e6
        
        d2g = d2g_coeff * ((d.r/au)**d2g_exp)

        # add N dust species
        ngrains = 10
        # agrains = np.array([5.e-5, 1.e-1])
        agrains = np.logspace(-5, -1, ngrains)


        a_max = np.array([])
        for rad_step in range(len(d.r)):
            a_max = np.append(a_max,np.max(agrains[agrains < (d.r[rad_step]/AU)**(-amax_exp) * amax_coeff]))

        a, a_i, sig_da = get_powerlaw_dust_distribution(d.sigma/100, a_max, q=4-size_exp, na=ngrains, a0=1e-5, a1=1e-1)
        # size_distri = agrains**size_exp
        # size_distri /= size_distri.sum()


        for eps, agrain in zip(np.transpose(sig_da),  a):
            d.add_dust(agrain=agrain, xigrain=rho_s, dtg=d2g * eps)

        # load the opacity from the previously calculated opacity table
        for dust in d.dust:
            # dust.grain.load_standard_opacity('draine2003', 'astrosilicates', verbose=True)
            dust.grain.read_opacity(os.path.join(temp_path, opac_fname + '.npz'))

        # compute the mean opacities
        d.meanopacitymodel = opac_params
        d.compute_mean_opacity()

        def movingaverage(interval, window_size):
            window = np.ones(int(window_size))/float(window_size)
            return np.convolve(interval, window, 'same')

        d.mean_opacity_planck[7:]    = movingaverage(d.mean_opacity_planck,10)[7:]
        d.mean_opacity_rosseland[7:] = movingaverage(d.mean_opacity_rosseland,10)[7:]

        for iter in range(100):
            d.compute_hsurf()
            d.compute_flareindex()
            d.compute_flareangle_from_flareindex(inclrstar=True)
            d.compute_disktmid(keeptvisc=False)
            d.compute_cs_and_hp()

        disk2d = disklab.Disk2D(disk=d, zrmax=zrmax,
                                meanopacitymodel=d.meanopacitymodel, nz=100)

        # snippet vertstruc 2d_1
        for vert in disk2d.verts:
            vert.iterate_vertical_structure()
        disk2d.radial_raytrace()
        for vert in disk2d.verts:
            vert.solve_vert_rad_diffusion()
            vert.tgas = (vert.tgas**4 + 15**4)**(1/4)
            for dust in vert.dust:
                dust.compute_settling_mixing_equilibrium()

        rmcd = disklab.radmc3d.get_radmc3d_arrays(disk2d, showplots=False)

        # Assign the radmc3d data

        # nphi = rmcd['nphi']
        ri = rmcd['ri']
        thetai = rmcd['thetai']
        phii = rmcd['phii']
        nr = rmcd['nr']
        # nth = rmcd['nth']
        # nphi = rmcd['nphi']
        rho = rmcd['rho']
        rmcd_temp = rmcd['temp']

        # Define the wavelength grid for the radiative transfer

        nlam = 200
        lam = np.logspace(-5, 1, nlam)
        lam_mic = lam*1e4

        # Write the `RADMC3D` input

        radmc3d.write_stars_input(d, lam_mic, path=temp_path)
        radmc3d.write_grid(ri, thetai, phii, mirror=False, path=temp_path)
        radmc3d.write_dust_density(rmcd_temp, fname = 'dust_temperature.dat', path=temp_path, mirror=False) # writes out the temperature
        radmc3d.write_dust_density(rho, mirror=False, path=temp_path)
        radmc3d.write_wavelength_micron(lam_mic, path=temp_path)
        radmc3d.write_opacity(disk2d, path=temp_path)
        radmc3d.write_radmc3d_input(
            {'scattering_mode': 5, 'scattering_mode_max': 5, 'nphot': 10000000},
            path=temp_path)

        # mm continuum image calculation
        lam_obs = 0.125
        rd = 300 * au
        radmc3d.radmc3d(
            f'image incl 47.5 posang -144.4 npix 500 lambda {lam_obs * 1e4} sizeau {4*rd/au} secondorder  setthreads 1',
            path=temp_path)

        # obtaining the resultant radial intensity profile and subsequent calculation of the logprob from this image divided by number of observation
        im = radmc3d.read_image(filename=os.path.join(temp_path, 'image.out'))
        radial_profile = []
        radial = []
        for x, y in zip(range(0, 251, 1), range(249, 500, 1)):
            radial_profile.append(
                im.image[y][int(round(249 + x * np.tan(0.6213372)))])
            radial.append(np.sqrt(
                (im.x[int(round(249 + x * np.tan(0.6213372)))] / au)**2 + (im.y[y] / au)**2))
        radial = np.asarray(radial)
        radial_profile = np.asarray(radial_profile)

        radial_profile   = radial_profile[radial>158]
        radial           = radial[radial>158]

        observed_intensity        = observed_intensity[observed_radius>1]
        observed_intensity_error  = observed_intensity_error[observed_radius>1]
        observed_radius           = observed_radius[observed_radius>1]


        log_prob_mm = -0.5 * np.sum((np.interp(observed_radius, radial / 158,
                                            radial_profile) - observed_intensity)**2 / (observed_intensity_error**2)) / len(observed_radius)

        #scattered light calculation
        #first the opacity matrices
        oc, rho_s = opacity.get_dsharp_mix()

        #a   = np.logspace(-5, -1, 100)
        m   = 4 * np.pi / 3 * rho_s * agrains**3
        nlam = 200
        lam = np.logspace(-5,1,nlam)

        res = opacity.get_opacities(agrains, lam, rho_s, oc, extrapolate_large_grains=False, n_angle=100)

        k_abs = res['k_abs']
        k_sca = res['k_sca']
        S1    = res['S1']
        S2    = res['S2']
        theta = res['theta']
        g     = res['g']

        # index where theta = 90 degree

        i90 = theta.searchsorted(90)

        zscat = opacity.calculate_mueller_matrix(lam, m, S1, S2, theta=theta, k_sca=k_sca)['zscat']

        chopforward = 5
        zscat_nochop = np.zeros((len(agrains) ,nlam, len(theta), 6))
        kscat_nochop = np.zeros((len(agrains), nlam))
        g_nochop     = np.zeros((len(agrains), nlam))

        for grain in range(len(agrains)):
            for i in range(nlam):
                #
                #
                # Now loop over the grain sizes
                #
                if chopforward > 0:
                    iang = np.where(theta < chopforward)
                    if theta[0] == 0.0:
                        iiang = np.max(iang) + 1
                    else:
                        iiang = np.min(iang) - 1
                    zscat_nochop[grain, i, :, :] = zscat[grain, i, :, :]  # Backup
                    kscat_nochop[grain, i] = k_sca[grain, i]      # Backup
                    g_nochop[grain, i]     = g[grain, i]
                    zscat[grain, i, iang, 0] = zscat[grain, i, iiang, 0]
                    zscat[grain, i, iang, 1] = zscat[grain, i, iiang, 1]
                    zscat[grain, i, iang, 2] = zscat[grain, i, iiang, 2]
                    zscat[grain, i, iang, 3] = zscat[grain, i, iiang, 3]
                    zscat[grain, i, iang, 4] = zscat[grain, i, iiang, 4]
                    zscat[grain, i, iang, 5] = zscat[grain, i, iiang, 5]
                    mu = np.cos(theta * np.pi / 180.)
                    dmu = np.abs(mu[1:len(theta)] - mu[0:(len(theta) - 1)])
                    zav = 0.5 * (zscat[grain, i, 1:len(theta), 0] + zscat[grain, i, 0:len(theta)-1, 0])
                    dum = 0.5 * zav * dmu
                    sum = dum.sum() * 4 * np.pi
                    k_sca[grain, i] = sum

                    mu_2 = 0.5 * (np.cos(theta[1:len(theta)] * np.pi / 180.) + np.cos(theta[0:len(theta)-1] * np.pi / 180.))
                    P_mu = 0.5 * ((2*np.pi*zscat[grain, i, 1:len(theta), 0] / k_sca[grain, i]) + (2*np.pi*zscat[grain, i, 0:len(theta)-1, 0] / k_sca[grain, i]))
                    g[grain,i] = np.sum(P_mu*mu_2*dmu)

        opacity_dict = {'a':agrains,
                'lam':lam,
                'theta':theta,
                'rho_s':rho_s,
                'k_abs':k_abs,
                'k_sca':k_sca,
                'g': g,
                'zscat':zscat}

        # add the scattering matrix to the dictionary
        res['zscat'] = zscat

        for filename in glob.glob(os.path.join(temp_path, "dustkappa_*.inp")):
            os.remove(filename)

        for x in agrains:
            i_grain = agrains.searchsorted(x)
            opacity.write_radmc3d_scatmat_file(i_grain, opacity_dict, f'{i_grain}', path=temp_path )

        def write(fid, *args, **kwargs):
            fmt = kwargs.pop('fmt', '')
            sep = kwargs.pop('sep', ' ')
            fid.write(sep.join([('{' + fmt + '}').format(a) for a in args]) + '\n')


        with open(os.path.join(temp_path, 'dustopac.inp'), 'w') as f:
            write(f, '2               Format number of this file')
            write(f, '{}              Nr of dust species'.format(len(agrains)))

            for x in agrains:
                i_grain = agrains.searchsorted(x)
                write(f, '============================================================================')
                write(f, '10               Way in which this dust species is read')
                write(f, '0               0=Thermal grain')
                write(f, '{}              Extension of name of dustscatmat_***.inp file'.format(i_grain))

            write(f, '----------------------------------------------------------------------------')

        #image calculation
        rd = 250 * au
        #radmc3d.radmc3d(f'image incl 47.5 posang -54.4 npix 500 lambda 1.65 sizeau 500 setthreads 4')
        radmc3d.radmc3d(f'image incl 47.5 posang 54.4 npix 500 lambda 1.65 sizeau 1000 setthreads 4',
            path=temp_path)


        #data = analyze.readData(dtemp=True, binary = False)
        im = image.readImage(os.path.join(temp_path, 'image.out'))
        im.writeFits(os.path.join(temp_path, 'image.fits'), dpc=158., coord='15h56m09.17658s -37d56m06.1193s')
        
        disk   = 'IMLup'
        #fname  = dh.get_datafile(disk)
        fname  = 'Qphi_IMLup.fits'
        PA     = dh.sources.loc[disk]['PA']
        inc    = dh.sources.loc[disk]['inc']
        d      = dh.sources.loc[disk]['distance [pc]']
        T_star = 10.**dh.sources.loc[disk]['log T_eff/ K']
        M_star = 10.**dh.sources.loc[disk]['log M_star/M_sun'] * c.M_sun.cgs.value
        L_star = 10.**dh.sources.loc[disk]['log L_star/L_sun'] * c.L_sun.cgs.value
        R_star = np.sqrt(L_star / (4 * np.pi * c.sigma_sb.cgs.value * T_star**4))

        clip =3.0

        # fix the header of the sphere image
        hdulist = fits.open(fname)
        hdu0 = hdulist[0]

        hdu0.header['cdelt1'] = -3.405e-06
        hdu0.header['cdelt2'] = 3.405e-06
        hdu0.header['crpix1'] = hdu0.header['naxis1']//2+1
        hdu0.header['crpix2'] = hdu0.header['naxis2']//2+1
        hdu0.header['crval1'] = 0.0
        hdu0.header['crval2'] = 0.0
        hdu0.header['crval3'] = 1.65e-4
        data = imagecube(hdulist, clip=clip)


        x, y, dy = data.radial_profile(inc=inc, PA=PA,z0=0.2, psi=1.27)

        profile = (y * u.Jy / data.beam_area_str).cgs.value
        profile_err = (dy * u.Jy / data.beam_area_str).cgs.value

        image_name = os.path.join(temp_path,'image.fits')

        # calculating the profile from scattered light image using imagecube and then calculation of the logprob
        sim_data = imagecube(image_name, clip=clip)

        sim_x, sim_y, sim_dy = sim_data.radial_profile(inc=inc, PA=PA, z0=0.2, psi=1.27)

        sim_profile     = (sim_y * u.Jy / sim_data.beam_area_str).cgs.value
        sim_profile_err = (sim_dy * u.Jy / sim_data.beam_area_str).cgs.value

        profile     = profile[x>1]
        profile_err = profile_err[x>1]
        x           = x[x>1]

        sim_profile     = sim_profile[sim_x>1]
        sim_profile_err = sim_profile_err[sim_x>1]
        sim_x           = sim_x[sim_x>1]

        log_prob_scat = -0.5 * np.nansum((np.interp(x,sim_x,sim_profile) - profile)**2 / (profile_err**2)) / len(x)

        # adding the two log probs and then multiplying by a large factor in order to make the MCMC more sensitive to changes
        log_prob = (log_prob_mm + log_prob_scat)*100 

    except:
        log_prob = -1e6
        
    return log_prob
