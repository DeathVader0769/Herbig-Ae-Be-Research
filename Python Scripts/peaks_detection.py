from pathlib import Path
import requests
import tqdm
import numpy as np
from io import BytesIO
from astropy.io import fits
import gzip
import winsound  # NOTE (point 7): Windows-only. If this ever runs on a remote/non-Windows
                  # machine, this import will fail -- swap for a cross-platform beep or drop it.
from scipy.signal import find_peaks
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib3.util.retry import Retry
import pandas as pd
import threading
import time

maxworker=100

lock = threading.Lock()

testing_obsid = pd.read_csv(r"E:\Python Works\University\Research\Data\Actual\obsids+rv.csv")


output_dir = Path(r"E:\Python Works\University\Research\Data\Actual\test_spectra")
output_dir.mkdir(parents=True, exist_ok=True)


testing_obsid_array=testing_obsid["obsid"].to_numpy()
testing_obsid_array=testing_obsid_array.flatten()


testing_obsid_rv_array=testing_obsid["rv"].to_numpy()
testing_obsid_rv_array=testing_obsid_rv_array.flatten()

bin=[]
width_not_enough_obsids=[]
width_enough_obsids=[]
no_peak_found_obsids=[]          # point 5: obsids that parsed fine but matched nothing anywhere
failure_log=[]                   # bin entries now carry a reason, not just an obsid

works=0
not_works=0
not_works_response_code=0
all_works=0
all_works_uncertain=0

# --- session with pooled connections + retry/backoff on transient server errors ---
# (fixes the "uncaught request exception kills the whole run" bug, and the backoff
# also acts as a soft politeness delay for point 8, see rate_limit() below too)
session=requests.Session()
retry_strategy=Retry(total=3,backoff_factor=1,status_forcelist=[429, 500, 502, 503, 504],allowed_methods=["GET"],)
adapter=requests.adapters.HTTPAdapter(max_retries=retry_strategy,pool_connections=maxworker,pool_maxsize=maxworker,)
session.mount("https://", adapter)
session.mount("http://", adapter)

# --- point 8: simple rate limiter shared across all worker threads ---
# Enforces a minimum gap between request *starts*, regardless of which thread
# fires it. This caps us at roughly 1/MIN_REQUEST_INTERVAL requests per second
# no matter how many workers are running, independent of maxworker. Tune this
# (and/or maxworker) according to LAMOST's API terms if you have them.
MIN_REQUEST_INTERVAL=0.01  # ~10 req/s ceiling across all 20 workers combined
rate_lock=threading.Lock()
_last_request_time=[0.0]

def rate_limit():
    with rate_lock:
        now = time.monotonic()
        wait = _last_request_time[0] + MIN_REQUEST_INTERVAL - now
        if wait > 0:
            time.sleep(wait)
        _last_request_time[0] = time.monotonic()


def peak_detector(id_obs,rv_val):
    global all_works
    global not_works
    global not_works_response_code
    global all_works_uncertain
    global works

    # point 2: resumability -- if we already saved this obsid on a prior run, skip
    # the network call entirely. Also sidesteps point 3 (write collisions) since a
    # given obsid is only ever submitted once per run, and now we short-circuit
    # before ever writing again if the file's already there.
    filename=str(id_obs)+".fits.gz"
    save_file_path=output_dir/filename
    if save_file_path.exists():
        with lock:
            all_works+=1
            width_enough_obsids.append(id_obs)
        return

    url=f"https://www.lamost.org/openapi/dr11/v2.0/lrs/spectrum/fits?obsid={id_obs}"

    rate_limit()  # point 8

    # request now wrapped in its own try/except: previously an uncaught
    # requests exception here propagated to future.result() in the main thread
    # and killed the entire batch -- this is why bin_obsids.txt differed
    # unpredictably between runs.
    try:
        # (connect_timeout, read_timeout) tuple instead of one shared timeout=20.
        # Note this still isn't a true wall-clock cap -- requests resets the read
        # timeout on every byte received -- but it bounds each stage far tighter
        # than before.
        # point 4: stream=False (unchanged) loads the whole body into memory before
        # any check happens. Fine at current file sizes; revisit if spectra sizes grow.
        response=session.get(url,stream=False,timeout=(10,15))
    except requests.exceptions.RequestException as e:
        with lock:
            bin.append(id_obs)
            failure_log.append((id_obs,"request_error",str(e)))
        return

    if response.status_code==200:
        file_type=response.headers.get("Content-Type")
        if file_type=="application/fits":
            temp_file=gzip.GzipFile(fileobj=BytesIO(response.content))
            try:
                with fits.open(temp_file) as hdul:
                    # point 1: works was previously incremented without a lock
                    with lock:
                        works+=1
                    found_peaks=find_peaks(hdul[1].data["NORMALIZATION"][0],prominence=0.2,width=(1,np.inf))
                    found_peaks_3=find_peaks(hdul[1].data["NORMALIZATION"][0],prominence=0.2,width=(3,np.inf))
                    specific_index_h_alpha=-9999
                    doppler_shifter_wavelength=6562.8*(1+((rv_val)/299792.458))
                    wv_max=doppler_shifter_wavelength+2
                    wv_min=doppler_shifter_wavelength-2

                    for peaks in found_peaks_3[0]:
                        peak_index_found=hdul[1].data["WAVELENGTH"][0][peaks]
                        if peak_index_found>=wv_min and peak_index_found<=wv_max:
                            specific_index_h_alpha=peaks
                            with lock:
                                width_enough_obsids.append(id_obs)
                            break

                    if specific_index_h_alpha!=-9999:
                        with save_file_path.open("wb") as file:
                            file.write(response.content)
                        # point 1: all_works previously incremented without a lock
                        with lock:
                            all_works+=1
                    else:
                        found_narrow=False
                        for peak in found_peaks[0]:
                            peak_index_found_no_width=hdul[1].data["WAVELENGTH"][0][peak]
                            if peak_index_found_no_width>=wv_min and peak_index_found_no_width<=wv_max:
                                with lock:
                                    width_not_enough_obsids.append(id_obs)
                                found_narrow=True
                                break
                        # point 5: previously, if neither loop matched, this obsid
                        # vanished from every list. Now it's tracked explicitly so
                        # len(width_enough)+len(width_not_enough)+len(no_peak_found)+len(bin)
                        # actually adds up to len(testing_obsid_array).
                        if not found_narrow:
                            with lock:
                                no_peak_found_obsids.append(id_obs)

            except Exception as e:
                with lock:
                    bin.append(id_obs)
                    failure_log.append((id_obs,"fits_parse_error",str(e)))
        else:
            # point 6: log the actual Content-Type instead of silently binning --
            # lets you tell "genuinely not a fits response" from "header quirk"
            with lock:
                bin.append(id_obs)
                failure_log.append((id_obs,"unexpected_content_type",str(file_type)))
    else:
        with lock:
            bin.append(id_obs)
            failure_log.append((id_obs,"bad_status_code",str(response.status_code)))
            # point 1: this counter was previously incremented outside any lock
            not_works_response_code+=1

print("Starting Threads")
with ThreadPoolExecutor(max_workers=maxworker) as executor:
    futures = {executor.submit(peak_detector, obsid, rv): obsid for obsid, rv in zip(testing_obsid_array, testing_obsid_rv_array)}
    for future in tqdm.tqdm(as_completed(futures), total=len(testing_obsid_array)):
        # belt-and-suspenders: if peak_detector somehow still raises outside its
        # own try/excepts, this stops it from taking down the whole run.
        try:
            future.result()
        except Exception as e:
            obsid=futures[future]
            with lock:
                bin.append(obsid)
                failure_log.append((obsid,"unexpected_error",str(e)))

with open(r"E:\Python Works\University\Research\Data\Actual\bin\bin_downloader\width_enough_obsids.txt", "a") as file:
    for obsid in width_enough_obsids:
        file.write(f"{obsid}\n")

with open(r"E:\Python Works\University\Research\Data\Actual\bin\bin_downloader\width_not_enough_obsids.txt", "a") as file:
    for obsid in width_not_enough_obsids:
        file.write(f"{obsid}\n")

with open(r"E:\Python Works\University\Research\Data\Actual\bin\bin_downloader\bin_obsids.txt", "a") as file:
    for obsid in bin:
        file.write(f"{obsid}\n")

# point 5: obsids that parsed cleanly but matched no peak anywhere
with open(r"E:\Python Works\University\Research\Data\Actual\bin_downloader\no_peak_found_obsids.txt", "a") as file:
    for obsid in no_peak_found_obsids:
        file.write(f"{obsid}\n")

with open(r"E:\Python Works\University\Research\Data\Actual\bin_downloader\bin_failure_log.csv", "a") as file:
    for obsid, reason, detail in failure_log:
        detail_clean = str(detail).replace(",", ";").replace("\n", " ")
        file.write(f"{obsid},{reason},{detail_clean}\n")

print(f"Finally Works={all_works}")
print(f"Bin (needs retry / investigate)={len(bin)}, No peak found={len(no_peak_found_obsids)}, Width not enough={len(width_not_enough_obsids)}")
print(f"Accounted for: {len(width_enough_obsids)+len(width_not_enough_obsids)+len(no_peak_found_obsids)+len(bin)} / {len(testing_obsid_array)}")

for i in range(100):
    winsound.Beep(2500, 750)
