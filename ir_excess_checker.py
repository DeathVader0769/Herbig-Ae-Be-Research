from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
import astropy.units as u
from dust_extinction.parameter_averages import F99
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import tqdm
import winsound
import warnings
from astropy.io.fits.verify import VerifyWarning


warnings.simplefilter('ignore', category=VerifyWarning)

maxworker=100
file_lock=threading.Lock()

csv_for_obsids_and_ra_and_dec=pd.read_csv(r"E:\Python Works\University\Research\Data\Actual\vizier_ra_and_dec_width_enough_with_lamost_obsids.csv")

ra_data=csv_for_obsids_and_ra_and_dec["ra"].to_numpy()
ra_data=ra_data.flatten()

dec_data=csv_for_obsids_and_ra_and_dec["dec"].to_numpy()
dec_data=dec_data.flatten()

lamost_obsids=csv_for_obsids_and_ra_and_dec["obsid"].to_numpy()
lamost_obsids=lamost_obsids.flatten()

sure_haebe=0

photometric_data_hdul_wise=fits.open(r"E:\Python Works\University\Research\Data\Actual\test_spectra\Wise Data\Wise_data_width_enough.fit")
photometric_data_hdul_2mass=fits.open(r"E:\Python Works\University\Research\Data\Actual\test_spectra\Wise Data\2mass_alone.fit")

lamost_coords=SkyCoord(ra=ra_data*u.deg,dec=dec_data*u.deg,frame="icrs")

twomass_ra=[]
twomass_dec=[]

for star in photometric_data_hdul_2mass[1:]:
    if star.data is not None and len(star.data)>0:
        twomass_ra.append(star.data["_RAJ2000"][0])
        twomass_dec.append(star.data["_DEJ2000"][0])

wise_ra=[]
wise_dec=[]

for star in photometric_data_hdul_wise[1:]:
    if star.data is not None and len(star.data)>0:
        wise_ra.append(star.data["_RAJ2000"][0])
        wise_dec.append(star.data["_DEJ2000"][0])

twomass_coords=SkyCoord(ra=twomass_ra*u.deg,dec=twomass_dec*u.deg,frame="icrs")
wise_coords=SkyCoord(ra=wise_ra*u.deg,dec=wise_dec*u.deg,frame="icrs")

idx_2mass,sep_2mass,_=lamost_coords.match_to_catalog_sky(twomass_coords)

idx_wise,sep_wise,_=lamost_coords.match_to_catalog_sky(wise_coords)

max_sep=2*u.arcsec

has_2mass=sep_2mass<max_sep
has_wise=sep_wise<max_sep

wv=[12350,16620,21590,34000,46000,120820,221940] #j,h,k,w1,w2,w3,w4

#print(has_2mass)

Fitzpatrick_model=F99(Rv=5)

def extinction_lada_indices(coordinate_lamost,bands,flag):

    gaia_source_id_temp=gaia_xmatch_for_source(coordinate_lamost)
    if gaia_source_id_temp is None:
        return False

    j_obs=bands[0]
    h_obs=bands[1]
    k_obs=bands[2]


    flux_j=(3e-8/(((wv[0])*u.AA).to(u.micron))**2)*(1594*10**(-0.4*j_obs))
    flux_h=(3e-8/(((wv[1])*u.AA).to(u.micron))**2)*(1024*10**(-0.4*h_obs))
    flux_k=(3e-8/(((wv[2])*u.AA).to(u.micron))**2)*(666.7*10**(-0.4*k_obs))


    hk_obs=h_obs-k_obs
    hk_error=extinction_corrector(gaia_source_id_temp,coordinate_lamost)

    if hk_error is None:
        return False


    hk_0=hk_obs-hk_error

    lada_index_2=None
    if flag=="Wise":
        w2_obs=bands[4]
        flux_w2=(3e-8/(((wv[4])*u.AA).to(u.micron))**2)*(171.787*10**(-0.4*w2_obs))
        lada_index_2=(np.log10((flux_w2*wv[4])/(flux_k*wv[2])))/(np.log10((wv[4])/(wv[2])))
    

    if hk_0>0.4:
        with file_lock:
            with open(r"E:\Python Works\University\Research\Data\Actual\bin\bin_ir_excess\CONFIRMED_HAEBE.txt", "a") as file:
                file.write(f"{obsid_locator(coordinate_lamost)}     HAeBe_HK    {hk_0}   {lada_index_2}     {coordinate_lamost.ra*u.deg}        {coordinate_lamost.dec*u.deg}\n")
        return True
    elif lada_index_2 is None:
        bin_writer(coordinate_lamost,"No Wise Data")
        return False 
    elif lada_index_2>-1.5:
        with file_lock:
            with open(r"E:\Python Works\University\Research\Data\Actual\bin\bin_ir_excess\CONFIRMED_HAEBE.txt", "a") as file:
                file.write(f"{obsid_locator(coordinate_lamost)}     HAeBe_Lada  {hk_0}   {lada_index_2}     {coordinate_lamost.ra*u.deg}        {coordinate_lamost.dec*u.deg}\n")
        return True
    else:
        return False

def gaia_xmatch_for_source(coordinates_of_source):

    #coordinates_of_source=SkyCoord(ra=ra*u.deg,dec=dec*u.deg,frame='icrs')
    gaia_xmatch=Gaia.cone_search(coordinates_of_source,radius=2*u.arcsec).get_results()
    if len(gaia_xmatch)==0:
        bin_writer(coordinates_of_source,"No Gaia Match")
        return None
    else:
        min_index_for_xmatch=np.argmin(gaia_xmatch["dist"])
        gaia_obsid=gaia_xmatch["source_id"][min_index_for_xmatch]
        return gaia_obsid

def extinction_corrector(gaia_obsid_received,lamost_coordinate):
    query_for_gaia=f"""
        SELECT
        source_id,
        teff_gspphot,
        azero_gspphot,
        azero_gspphot_lower,
        azero_gspphot_upper,
        ag_gspphot,
        ebpminrp_gspphot,
        libname_gspphot
        FROM gaiadr3.gaia_source
        WHERE source_id = {gaia_obsid_received}
        """
    gaia_fetched_value=Gaia.launch_job(query_for_gaia).get_results()
    if len(gaia_fetched_value)==0:
        bin_writer(lamost_coordinate,"No A0")
        return None
    A0_gaia=gaia_fetched_value["azero_gspphot"][0]

    if not np.isfinite(A0_gaia):
        bin_writer(lamost_coordinate,"A0 infinite")
        return None
    wavelength_for_bands=[0.5414*u.micron,(wv[1]*u.AA).to(u.micron),(wv[2]*u.AA).to(u.micron)]
    A0_AV=Fitzpatrick_model(1/wavelength_for_bands[0])
    AH_AV=Fitzpatrick_model(1/(wavelength_for_bands[1]))
    AK_AV=Fitzpatrick_model(1/(wavelength_for_bands[2]))

    AV=A0_gaia/A0_AV
    extinction_Correction=AV*((AH_AV)-((AK_AV)))


    return extinction_Correction

def obsid_locator(coordinates):
    
    idx,sep,_=coordinates.match_to_catalog_sky(lamost_coords)

    obsiddd=lamost_obsids[idx]
    return obsiddd

def bin_writer(coordinates_to_be_binned,cause):

    obsid_to_be_binned=obsid_locator(coordinates_to_be_binned)

    with file_lock:
        with open(r"E:\Python Works\University\Research\Data\Actual\bin\bin_ir_excess\bin_obsids.txt", "a") as file:
            file.write(f"{obsid_to_be_binned}     {cause}\n")


fl=""
band=[]
futures=[]

sure_haebe=0

with ThreadPoolExecutor(max_workers=maxworker) as executor:
    for c,coordinate in enumerate(lamost_coords):
        if has_wise[c]==True:
            j=(photometric_data_hdul_wise[idx_wise[c]+1].data["Jmag"]).flatten()[0]
            h=(photometric_data_hdul_wise[idx_wise[c]+1].data["Hmag"]).flatten()[0]
            k=(photometric_data_hdul_wise[idx_wise[c]+1].data["Kmag"]).flatten()[0]

            w1=(photometric_data_hdul_wise[idx_wise[c]+1].data["W1mag"]).flatten()[0]
            w2=(photometric_data_hdul_wise[idx_wise[c]+1].data["W2mag"]).flatten()[0]
            w3=(photometric_data_hdul_wise[idx_wise[c]+1].data["W3mag"]).flatten()[0]
            w4=(photometric_data_hdul_wise[idx_wise[c]+1].data["W4mag"]).flatten()[0]

            band=[j,h,k,w1,w2,w3,w4]

            fl="Wise"
        elif has_2mass[c]==True:
            j=(photometric_data_hdul_2mass[idx_2mass[c]+1].data["Jmag"]).flatten()[0]
            h=(photometric_data_hdul_2mass[idx_2mass[c]+1].data["Hmag"]).flatten()[0]
            k=(photometric_data_hdul_2mass[idx_2mass[c]+1].data["Kmag"]).flatten()[0]

            band=[j,h,k]

            fl="2mass"
        else:
            continue
        future=executor.submit(extinction_lada_indices,coordinate,band,fl)
        futures.append(future)


    for future in tqdm.tqdm(as_completed(futures),total=len(futures),desc="Processing Stars"):
        if future.result():
            sure_haebe+=1

for i in range(5):
    winsound.Beep(2500,500)
