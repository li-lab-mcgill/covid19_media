####Open csv file#####
####Read source (link) as the input from domain names####

####Get the country code from the domain####
import socket
from geolite2 import geolite2

def origin(ip, domain_str, result):
    print("{0} [{1}]: {2}".format(domain_str.strip(), ip, result))
    

def getip(domain_str):
    ip = socket.gethostbyname(domain_str.strip())
    reader = geolite2.reader()      
    output = reader.get(ip)
    result = output['country']['iso_code']
    origin(ip, domain_str, result)

with open("/path/to/hostnames.txt", "r") as ins:
    for domain_str in ins:
        try:
            getip(domain_str)
        except socket.error as msg:
            print("{0} [could not resolve]".format(domain_str.strip())) 
            if len(domain_str) > 2:
                subdomain = domain_str.split('.', 1)[1]
                try:
                    getip(subdomain)
                except:
                    continue

geolite2.close()

####Add column with the country codes####

