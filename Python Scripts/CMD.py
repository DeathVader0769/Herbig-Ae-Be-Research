import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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
import ezpadova

plt.style.use('seaborn-v0_8-darkgrid')


warnings.simplefilter('ignore', category=VerifyWarning)

Fitzpatrick_model=F99(Rv=3.1)

maxworker=100
file_lock=threading.Lock()

broad_line_ra_and_dec=pd.read_csv(r"E:\Python Works\University\Research\Data\Actual\vizier_ra_and_dec_width_enough_with_lamost_obsids.csv")

broadline_ra_data=broad_line_ra_and_dec["ra"].to_numpy().flatten()
broadline_dec_data=broad_line_ra_and_dec["dec"].to_numpy().flatten()
broadline_lamost_obsids=broad_line_ra_and_dec["obsid"].to_numpy().flatten()

narrowline_ra_and_dec_data=pd.read_csv(r"E:\Python Works\University\Research\Data\Actual\vizier table for ra and dec width not enough.csv")

narrowline_ra_data=narrowline_ra_and_dec_data["ra"].to_numpy().flatten()
narrowline_dec_data=narrowline_ra_and_dec_data["dec"].to_numpy().flatten()
narrowline_lamost_obsids=narrowline_ra_and_dec_data["obsid"].to_numpy().flatten()


broad_line_coords=SkyCoord(ra=broadline_ra_data*u.deg,dec=broadline_dec_data*u.deg,frame="icrs")
narrow_line_coords=SkyCoord(ra=narrowline_ra_data*u.deg,dec=narrowline_dec_data*u.deg,frame="icrs")

max_sep=2*u.arcsec

narrowline=[]
broadline=[]



def gaia_xmatch_for_source(coordinates_of_source):

    #coordinates_of_source=SkyCoord(ra=ra*u.deg,dec=dec*u.deg,frame='icrs')
    gaia_xmatch=Gaia.cone_search(coordinates_of_source,radius=2*u.arcsec).get_results()
    if len(gaia_xmatch)==0:
        return None
    else:
        min_index_for_xmatch=np.argmin(gaia_xmatch["dist"])
        gaia_obsid=gaia_xmatch["source_id"][min_index_for_xmatch]
        return gaia_obsid

def extinction_corrector(coordinate):

    gaia_obsid_received=gaia_xmatch_for_source(coordinate)

    if gaia_obsid_received is None:
       return None

    query_for_gaia=f"""
        SELECT
        source_id,
        ra,
        dec,
        phot_g_mean_mag,
        phot_bp_mean_mag,
        phot_rp_mean_mag,
        parallax,
        parallax_error,
        azero_gspphot,
        azero_gspphot_lower,
        azero_gspphot_upper,
        ebpminrp_gspphot,
        ag_gspphot,
        teff_gspphot
    FROM gaiadr3.gaia_source
    WHERE source_id = {gaia_obsid_received}
    """

    
    gaia_fetched_value=Gaia.launch_job(query_for_gaia).get_results()
    parallax=gaia_fetched_value["parallax"][0]
    bp_mag=gaia_fetched_value["phot_bp_mean_mag"][0]
    rp_mag=gaia_fetched_value["phot_rp_mean_mag"][0]
    ebp_rp_gaia=gaia_fetched_value["ebpminrp_gspphot"][0]
    ag_gaia=gaia_fetched_value["azero_gspphot"][0]
    if not (np.isfinite(parallax) and parallax>0 and np.isfinite(bp_mag) and np.isfinite(rp_mag) and np.isfinite(ebp_rp_gaia) and np.isfinite(ag_gaia)):
        return None
    bp_rp_0=(bp_mag-rp_mag)-ebp_rp_gaia
    a_rp=0.69*(ag_gaia/0.85)
    rp_0=rp_mag-a_rp
    m_rp_0=rp_0+5*np.log10(parallax)-10

    return m_rp_0,bp_rp_0

narrowline_x=[]  
narrowline_y=[]

with ThreadPoolExecutor(max_workers=maxworker) as executor:
  future_to_coord={executor.submit(extinction_corrector, coord): coord for coord in narrow_line_coords}

  for future in tqdm.tqdm(as_completed(future_to_coord),total=len(future_to_coord),desc="Processing Gaia X-match Narrow-Line"):
    result=future.result()

    if result is not None:
      m_rp_0,bp_rp_0=result
      narrowline_x.append(bp_rp_0) 
      narrowline_y.append(m_rp_0)

x_narrowline=np.array(narrowline_x)
y_narrowline=np.array(narrowline_y)

broadline_x=[]
broadline_y=[]


with ThreadPoolExecutor(max_workers=maxworker) as executor:
  future_to_coord={executor.submit(extinction_corrector, coord): coord for coord in broad_line_coords}

  for future in tqdm.tqdm(as_completed(future_to_coord),total=len(future_to_coord),desc="Processing Gaia X-match Broad-Match"):
    result=future.result()

    if result is not None:
      m_rp_0,bp_rp_0=result
      broadline_x.append(bp_rp_0) 
      broadline_y.append(m_rp_0)

x_broadline=np.array(broadline_x)
y_broadline=np.array(broadline_y)

iso=ezpadova.parsec.get_isochrones(logage=(7.0, 7.0, 1.0),MH=(0.0, 0.0, 0.1),photsys_file='gaiaEDR3')
iso_bp=iso['G_BPmag']
iso_rp=iso['G_RPmag']
iso_color=iso_bp - iso_rp
iso_m_rp=iso['G_RPmag']


plt.figure(figsize=(10,6))
plt.scatter(x_broadline,y_broadline,marker="*",label="Broad - Line",color="red",s=30)
plt.scatter(x_narrowline,y_narrowline,marker="o",label="Narrow - Line",color="yellow",s=20)
sort_idx = np.argsort(iso['Mini'])
plt.plot(iso_color[sort_idx],iso_m_rp[sort_idx],color='black',linewidth=1,label='10 Myr Isochrone',alpha=0.8,linestyle='--')
plt.grid(True,alpha=0.2,linestyle=":")
plt.gca().invert_yaxis()
plt.xlabel(r"$(BP - RP)_0$")
plt.ylabel(r"$M_{RP,0}$")
plt.title("Extinction Corrected Color-Magnitude Diagram")
plt.legend()
plt.savefig(r"Extinction_Corrected_CMD.svg",format="svg",bbox_inches="tight")
plt.close()

for i in range(3):
    winsound.Beep(1000, 500)
