####Open csv file#####
####Read source (link) as the input from domain names####

####Get the country code from the domain####
import socket
from geolite2 import geolite2
import pandas as pd 

array_country = []

#Get csv file
df = pd.read_csv("who_all.csv")

def origin(ip, domain_str, result):
    origin_country = "{0} [{1}]: {2}".format(domain_str.strip(), ip, result)
    print(origin_country)
    #Add to array
    array_country.append(origin_country)


def getip(domain_str):
    ip = socket.gethostbyname(domain_str.strip())
    reader = geolite2.reader()      
    output = reader.get(ip)
    result = output['country']['iso_code']
    origin(ip, domain_str, result)

ins = df["LINK"].tolist()
for domain_str in ins:
    try:
        domain = str(domain_str)
        char20 = domain[0:40]
        strip_char = '/'
        char20 = strip_char.join(char20.split(strip_char)[:3])
        char20 = char20[char20.find('www'):]
        getip(char20)
    except socket.error as msg:
        print("{0} [could not resolve]".format(char20.strip())) 
        if len(char20) > 2:
            try:
                subdomain = char20.split('.', 1)[1]
            except:
                continue
            try:
                getip(subdomain)
            except:
                continue

#Array to csv column : 
mapping = dict(enumerate(array_country))
df['C_ORIGIN'] = df['SOURCE'].map(mapping)
df.to_csv("who_country_origin.csv", index=False)

geolite2.close()

