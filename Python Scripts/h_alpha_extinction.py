import time
start_time=time.perf_counter()
from pathlib import Path
import requests
import tqdm
import numpy as np
from io import BytesIO
from astropy.io import fits
import gzip
import winsound
from scipy.signal import find_peaks,peak_widths
from concurrent.futures import ThreadPoolExecutor,as_completed
from urllib3.util.retry import Retry
import pandas as pd
import threading
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.io.fits.verify import VerifyWarning
from dust_extinction.parameter_averages import F99
from requests.adapters import HTTPAdapter
from astropy.coordinates import SkyCoord
import astropy.units as u 
import warnings
import csv

plt.style.use('dark_background')

Fitzpatrick_model=F99(Rv=3.1)

maxworker=200
file_lock=threading.Lock()


bin_file=r"E:\Python Works\University\Research\Data\Actual\ver 2\bin\halpha_extinction\restimation version\bin_full_200_workers.csv"
bin_file_path=Path(bin_file)
bin_file_path.unlink(missing_ok=True)
bin_file_path.touch()


wv=[12350,16620,21590]

session=requests.sessions.Session()
retry=Retry(total=2,backoff_factor=0.5,status_forcelist=[429,500,502,503,504],allowed_methods=["GET"])
session.mount("https://",HTTPAdapter(max_retries=retry,pool_connections=maxworker,pool_maxsize=maxworker))
session.mount("http://",HTTPAdapter(max_retries=retry,pool_connections=maxworker,pool_maxsize=maxworker))

MIN_REQUEST_INTERVAL=0.005  # ~10 req/s ceiling across all 20 workers combined
rate_lock=threading.Lock()
_last_request_time=[0.0]

def rate_limit():
    with rate_lock:
        now = time.monotonic()
        wait = _last_request_time[0] + MIN_REQUEST_INTERVAL - now
        if wait > 0:
            time.sleep(wait)
        _last_request_time[0] = time.monotonic()



def estimate_spectral_type(spectrum, wavelength, l_classifier_corrected):
    CLASSIFIER_HALFWIDTH = [4.0, 4.0, 6.0, 6.0, 4.0, 6.0]
    THRESH = {"heII": 0.05, "heI": 0.05, "balmer": 0.25, "caK": 0.15, "G": 0.10}
    depths = []
    for center, hw in zip(l_classifier_corrected, CLASSIFIER_HALFWIDTH):
        window = np.abs(wavelength - center) <= hw
        if window.sum() < 2:
            depths.append(np.nan)          # line outside coverage
        else:
            depths.append(1.0 - np.nanmedian(spectrum[window]))

    if np.any(np.isnan(depths)):
        return "Unclassified", depths

    heII, heI, hd, hg, caK, g = depths
    balmer = (hd + hg) / 2

    if heII > THRESH["heII"]:
        stype = "O"
    elif heI > THRESH["heI"]:
        stype = "B"
    elif balmer > THRESH["balmer"] and caK < THRESH["caK"] and g < THRESH["G"]:
        stype = "A"
    elif caK >= THRESH["caK"] or g >= THRESH["G"]:
        stype = "Late"
    else:
        stype = "Uncertain"
    return stype, depths



def get_spectrum(id_obs):
    url=f"https://www.lamost.org/openapi/dr11/v2.0/lrs/spectrum/fits?obsid={id_obs}"
    rate_limit()
    try:
        getting_spectrum=session.get(url=url,timeout=(5,15),stream=False)
        getting_spectrum.raise_for_status()

        file_type=getting_spectrum.headers.get("Content-Type")
        if not file_type=="application/fits":
            return None,None,"Wrong File Type"

        temp_file=gzip.GzipFile(fileobj=BytesIO(getting_spectrum.content))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", VerifyWarning)

            with fits.open(temp_file) as spectrum_fits:
                norm_spectrum=spectrum_fits[1].data["NORMALIZATION"][0]
                wavelengths=spectrum_fits[1].data["WAVELENGTH"][0]

        code="File Works"

        #process_spectrum(norm_spectrum,wavelengths,2) #comment this once done

        return norm_spectrum,wavelengths,code #once done uncomment this
    
    except Exception as e:
        return None,None,str(e)


def doppler_shift_adjuster(rv):


    l_forbidden=[6548,6584,6717,6731,6300,6363]#add forbidden lines
    l_classifier=[4685.7,4471.5,4101.7,4340.5,3933.7,4300.0]#he2 , he1, hdel, hgama, weak Ca II K and G band 

    corrected_halpha=6562.8*(1+(rv/(299792.458)))
    l_forbidden_corrected=[x*(1+(rv/(299792.458))) for x in l_forbidden]
    l_classifier_corrected=[x*(1+(rv/(299792.458))) for x in l_classifier]

    return l_forbidden_corrected,corrected_halpha,l_classifier_corrected
 

def process_spectrum(spectrum,wavelength,radv):
    
    if (spectrum is not None) and (wavelength is not None):
        try: 
            max_spectrum=np.max(spectrum)
            peaks=find_peaks(x=spectrum,prominence=(0.2,max_spectrum),width=1)
            peaks_indices=peaks[0]

            peak_wavelengths=wavelength[peaks_indices]
            peak_wavelengths=peak_wavelengths[:,np.newaxis]
            forbidden_wavelengths,doppler_corrected_halpha,classifier_wavelengths=doppler_shift_adjuster(radv)

            spectral_type,depths_list=estimate_spectral_type(spectrum,wavelength,classifier_wavelengths)

            forbidden_wavelengths_h_alpha=np.array([doppler_corrected_halpha]+list(forbidden_wavelengths))
            diff_matrix=np.abs(peak_wavelengths-forbidden_wavelengths_h_alpha)

            halpha_match=diff_matrix[:,0]<=30

            forbidden_distances=diff_matrix[:,1:]
            forbidden_match=forbidden_distances<=20

            OI_match_upper=diff_matrix[:,-1]<=20
            OI_match_lower=diff_matrix[:,-2]<=20

            if spectral_type not in ["O","B","A","Uncertain"]:
                return None,None,None,None,None,None,None,spectral_type,depths_list

            
            if np.any(halpha_match):
                halpha_peak_indices=peaks_indices[halpha_match]
                halpha_distances=np.abs(peak_wavelengths[halpha_match,0]-doppler_corrected_halpha)
                closest_halpha=np.argmin(halpha_distances)
                halpha_peak_index=halpha_peak_indices[closest_halpha]
                halpha_peak_wavelength=wavelength[halpha_peak_index]
                widths,width_heights,left_ips,right_ips=peak_widths(spectrum,[halpha_peak_index],rel_height=0.5)
                peak_width_pixels=widths[0]
                left_index=max(0, int(np.floor(left_ips[0])))
                right_index=min(len(wavelength)-1,int(np.ceil(right_ips[0])))
                fwhm_angstrom=wavelength[right_index]-wavelength[left_index]
                peak_number=np.where(peaks_indices==halpha_peak_index)[0][0]
                halpha_prominence=peaks[1]["prominences"][peak_number]
                #print("Found") #comment this

                if np.any(forbidden_match):
                    return True,True,halpha_peak_wavelength,fwhm_angstrom,peak_width_pixels,halpha_prominence,"Worked",spectral_type,depths_list #uncomment
                else:
                    return True,False,halpha_peak_wavelength,fwhm_angstrom,peak_width_pixels,halpha_prominence,"Worked",spectral_type,depths_list #uncomment

            
            elif np.any(OI_match_lower) or np.any(OI_match_upper):
                return False,True,None,None,None,None,"OI Detected",spectral_type,depths_list
            elif np.any(forbidden_match):
                #print("Forbidden line detected") #comment
                return False,True,None,None,None,None,"Worked",spectral_type,depths_list
            else:
                #print("No halpha") #comment
                return False,None,None,None,None,None,"No halpha/no peaks deteced",spectral_type,depths_list #uncomment
 
        except Exception as e:
            return None,None,None,None,None,None,str(e),"Unclassified",[np.nan]*6
    else:
        return None,None,None,None,None,None,"Spectrum Invalid","Unclassified",[np.nan]*6

def extinction_corrector(j,h,k,a0,bp_mag,rp_mag,parallax):

    A0_gaia=a0
    wavelength_for_bands=[0.5414*u.micron,(wv[0]*u.AA).to(u.micron),(wv[1]*u.AA).to(u.micron),(wv[2]*u.AA).to(u.micron)]
    A0_AV=Fitzpatrick_model(1/wavelength_for_bands[0])
    AJ_AV=Fitzpatrick_model(1/wavelength_for_bands[1])
    AH_AV=Fitzpatrick_model(1/(wavelength_for_bands[2]))
    AK_AV=Fitzpatrick_model(1/(wavelength_for_bands[3]))
    AV=A0_gaia/A0_AV
    extinction_Correction=AV*((AH_AV)-((AK_AV)))



    j_extinction_corrected=j-(AV*AJ_AV)
    h_extinction_corrected=h-(AV*AH_AV)
    k_extinction_corrected=k-(AV*AK_AV)

    flux_j=((3e-8/(((wv[0])*u.AA).to(u.micron))**2)*(1594*10**(-0.4*j_extinction_corrected))).value
    flux_h=((3e-8/(((wv[1])*u.AA).to(u.micron))**2)*(1024*10**(-0.4*h_extinction_corrected))).value
    flux_k=((3e-8/(((wv[2])*u.AA).to(u.micron))**2)*(666.7*10**(-0.4*k_extinction_corrected))).value

    fluxes=np.array([flux_j,flux_h,flux_k])

    hk_0_0=(h-k)-extinction_Correction
    pot_her=False

    if hk_0_0>0.4:
        pot_her=True


    return fluxes,extinction_Correction,hk_0_0,pot_her,A0_gaia,AV,AJ_AV,AH_AV,AK_AV,bp_mag,rp_mag,parallax



def main(obsid,coordinates,twomass_photometric_data,radial_vel,sptype,A0,bp,rp,plx,ej,eh,ek,ebp,erp):
    
    spectra,lambdas,status_code=get_spectrum(id_obs=obsid)
    if status_code!="File Works":
        l2=[]
        l2=[obsid,None,None,None,None,None,None,None,None,None,None,"Download Failed",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]
        return l2

    halpha,forbidden,halpha_peak_wavelength,fwhm_angstrom,peak_width_pixels,halpha_prominence,code_processor,resptype,dep=process_spectrum(spectrum=spectra,wavelength=lambdas,radv=radial_vel)
    twomassj=twomass_photometric_data[0]
    twomassh=twomass_photometric_data[1]
    twomassk=twomass_photometric_data[2]


    if halpha is True or code_processor=="OI Detected":
        flux_jhk,hk_extinction_correction,hk_0,potential_herbig,gaia_A0,calc_AV,AJ_AV_,AH_AV_,AK_AV_,gaia_bp,gaia_rp,gaia_parallax=extinction_corrector(j=twomassj,h=twomassh,k=twomassk,parallax=plx,bp_mag=bp,rp_mag=rp,a0=A0)

        if flux_jhk is None:
            flux_jhk=[None,None,None]

        l=[obsid,coordinates.ra.deg,coordinates.dec.deg,sptype,resptype,dep[0],dep[1],dep[2],dep[3],dep[4],dep[5],status_code,twomassj,twomassh,twomassk,ej,eh,ek,flux_jhk[0],flux_jhk[1],flux_jhk[2],halpha,halpha_peak_wavelength,fwhm_angstrom,peak_width_pixels,halpha_prominence,forbidden,code_processor,gaia_parallax,gaia_A0,calc_AV,AJ_AV_,AH_AV_,AK_AV_,gaia_bp,gaia_rp,ebp,erp,hk_extinction_correction,hk_0,potential_herbig]
        return l
    else:
        return None


lamost_file=pd.read_csv(r"E:\Python Works\University\Research\Data\Actual\ver 2\20260924185822.csv")
lamost_obsids=((lamost_file["obsid"]).to_numpy()).flatten()
lamost_ra_data=((lamost_file["ra"]).to_numpy()).flatten()
lamost_dec_data=((lamost_file["dec"]).to_numpy()).flatten()
lamost_rv_data=(lamost_file["rv"].to_numpy()).flatten()
lamost_sp_type=(lamost_file["subclass"].to_numpy()).flatten()
lamost_coordinates=SkyCoord(ra=lamost_ra_data*u.deg,dec=lamost_dec_data*u.deg,frame="icrs")


twomass_file=pd.read_csv(r"E:\Python Works\University\Research\Data\Actual\ver 2\2mass_xmatch_full.csv")
twomass_ra_data=((twomass_file["ra"]).to_numpy()).flatten()
twomass_dec_data=((twomass_file["dec"]).to_numpy()).flatten()
twomass_j_data=((twomass_file["Jmag"]).to_numpy()).flatten()
twomass_h_data=((twomass_file["Hmag"]).to_numpy()).flatten()
twomass_k_data=((twomass_file["Kmag"]).to_numpy()).flatten()
twomass_ej_data=((twomass_file["e_Jmag"]).to_numpy()).flatten()
twomass_eh_data=((twomass_file["e_Hmag"]).to_numpy()).flatten()
twomass_ek_data=((twomass_file["e_Kmag"]).to_numpy()).flatten()
twomass_coordinates=SkyCoord(ra=twomass_ra_data*u.deg,dec=twomass_dec_data*u.deg,frame="icrs")

idx,d2d,_=lamost_coordinates.match_to_catalog_sky(twomass_coordinates)


gaia_file=pd.read_csv(r"E:\Python Works\University\Research\Data\Actual\ver 2\gaia_xmatch_full_csv.csv")
gaia_ra_data=gaia_file["ra"].to_numpy().flatten()
gaia_dec_data=gaia_file["dec"].to_numpy().flatten()
gaia_bp_data=gaia_file["BPmag"].to_numpy().flatten()
gaia_rp_data=gaia_file["RPmag"].to_numpy().flatten()
gaia_A0_data=gaia_file["A0"].to_numpy().flatten()
gaia_ebp_data=gaia_file["e_BPmag"].to_numpy().flatten()
gaia_erp_data=gaia_file["e_RPmag"].to_numpy().flatten()
gaia_parallax_data=gaia_file["Plx"].to_numpy().flatten()
gaia_rv_data=gaia_file["RV"].to_numpy().flatten()
gaia_coordinates=SkyCoord(ra=gaia_ra_data*u.deg,dec=gaia_dec_data*u.deg,frame="icrs")

idx_gaia,d2d_gaia,_=lamost_coordinates.match_to_catalog_sky(gaia_coordinates)


headers=["lamost_obsid","RA","DEC","LAMOST Spectral Type","Re-estimated Spectral Type","He II","He I","Hdelta","Hgamma","Ca II K","Ca II G","Download Spectra Code","Jmag","Hmag","Kmag","e_Jmag","e_Hmag","e_Kmag","JFlux","HFlux","KFlux","Halpha","Halpha Peak Wavelength","FWHM","Peak Width","Prominence","Forbidden Line","Code Processor","Parallax","A0","AV","AJ_AV","AH_AV","AK_AV","BP","RP","e_BPmag","e_RPmag","E(H-K)","(H-K)_0","Verdict on Herbig","error_message"]

with open(bin_file,"a",newline="") as temp:
    writer=csv.writer(temp)
    writer.writerow(headers)

print("Just before")
l1=[100000*x for x in range(25)]

with ThreadPoolExecutor(max_workers=maxworker) as executor:
    futures=[]
    for c,obs in enumerate(lamost_obsids):
        if d2d[c]<=2*u.arcsec and d2d_gaia[c]<=2*u.arcsec:
            photomet_data=[twomass_j_data[idx[c]],twomass_h_data[idx[c]],twomass_k_data[idx[c]]]
            lamost_rv=lamost_rv_data[c]
            azero=gaia_A0_data[idx_gaia[c]]
            gaia_rv=gaia_rv_data[idx_gaia[c]]
            rv=gaia_rv
            if not np.isfinite(azero):
                continue
            if not np.isfinite(gaia_rv):
                if np.isfinite(lamost_rv):
                    rv=lamost_rv   
                else:
                    rv=0        
            if c in l1:
                print("hmm")
                                                           
            futures.append(executor.submit(main,ebp=gaia_ebp_data[idx_gaia[c]],erp=gaia_erp_data[idx_gaia[c]],ej=twomass_ej_data[idx[c]],eh=twomass_eh_data[idx[c]],ek=twomass_ek_data[idx[c]],coordinates=lamost_coordinates[c],obsid=obs,twomass_photometric_data=photomet_data,radial_vel=rv,sptype=lamost_sp_type[c],bp=gaia_bp_data[idx_gaia[c]],rp=gaia_rp_data[idx_gaia[c]],plx=gaia_parallax_data[idx_gaia[c]],A0=gaia_A0_data[idx_gaia[c]]))
        else:
            continue

    for future in tqdm.tqdm(as_completed(futures),total=len(futures),desc="Processing Stars"):

        try:
            result=future.result()

            if result is not None:
                result_list=list(result)
                result_list.append("None")
                with file_lock:
                    with open(bin_file,"a",newline="") as temp:
                        writer=csv.writer(temp)
                        writer.writerow(result_list)   

        except Exception as e:
            with file_lock:
                with open(bin_file,"a",newline="") as temp:
                    writer=csv.writer(temp)
                    l=[None for i in range(len(headers))]
                    l[-1]=str(e)
                    writer.writerow(l)

print("Done")

for i in range(10):
    winsound.Beep(1000,500)

end_time=time.perf_counter()

print(f"Time Taken={end_time-start_time:.4f} Seconds")