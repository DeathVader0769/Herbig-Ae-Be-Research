from astropy.table import Table
from astroquery.xmatch import XMatch
import astropy.units as u

entries=["simbad","vizier:J/AJ/118/1043/stars","vizier:J/A+AS/104/315/liste3","vizier:J/A+AS/136/429/table1","vizier:J/A+A/638/A21/pms","vizier:J/AJ/129/856/table2","vizier:J/A+A/638/A21/cbe","vizier:J/A+A/620/A128/hqsample","vizier:J/A+A/620/A128/lqsample","vizier:J/ApJS/259/38/tablea1","vizier:J/ApJS/259/38/table2","vizier:J/A+AS/104/315/table1","vizier:J/A+AS/104/315/table6","vizier:J/A+AS/104/315/table4a","vizier:J/A+AS/104/315/table4b","vizier:J/ApJ/653/657/main","vizier:III/67A/catalog","vizier:J/AJ/157/159/table1","vizier:J/MNRAS/446/274/catalog"]

width_enough=Table.read(r"E:\Python Works\University\Research\Data\Actual\bin\bin_ir_excess\CONFIRMED_HAEBE_width_enough - Copy.csv")
width_not_enough=Table.read(r"E:\Python Works\University\Research\Data\Actual\bin\bin_ir_excess\CONFIRMED_HAEBE_width_not_enough - Copy.csv")

l_narrowline=[]
l_broadline=[]

for catalog in entries:
    print("*"*100)
    print(f"Cross-matching with {catalog}...")
    Xmatch_width_enough=XMatch.query(cat1=width_enough,cat2=catalog,max_distance=2*u.arcsec,colRA1='ra',colDec1='dec')
    Xmatch_width_not_enough=XMatch.query(cat1=width_not_enough,cat2=catalog,max_distance=2*u.arcsec,colRA1='ra',colDec1='dec')

    
    print(f"Found {len(Xmatch_width_enough)} matches with sufficient width.")
    print(f"Found {len(Xmatch_width_not_enough)} matches with insufficient width.")

    if len(Xmatch_width_enough)==0 and len(Xmatch_width_not_enough)==0:
        continue

    if len(Xmatch_width_enough)>0 and catalog=="simbad":
        print(Xmatch_width_enough["ra", "dec",'main_id', 'otype', 'nbref'])
        print("|"*100)
    else:
        print(Xmatch_width_enough["ra", "dec"])
        print("|"*100)
    if len(Xmatch_width_not_enough)!=0 and catalog=="simbad":
        print(Xmatch_width_not_enough["ra", "dec", 'main_id', 'otype', 'nbref'])
        print("|"*100)
    else:
        print(Xmatch_width_not_enough["ra", "dec"])
        print("|"*100)
    
    #print(Xmatch_width_not_enough["ra", "dec", 'main_id', 'otype', 'nbref'])

    l_narrowline.append(Xmatch_width_not_enough["ra","dec"])
    l_broadline.append(Xmatch_width_enough["ra","dec"])

print("-"*100)
print("Cross-matching complete.")
print(f"Total narrow-line matches: {len(l_narrowline)}")
print(l_narrowline)
print("-"*100)
print(f"Total broad-line matches: {len(l_broadline)}")
print(l_broadline)
