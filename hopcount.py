import concurrent.futures
from socket import timeout
from numpy.lib.function_base import average
import pandas as pd
from scapy.all import IP, traceroute
import tldextract
data = pd.read_csv("tmp.csv")[:10000]
data = data[['url', 'label']]
mp = {}
mmp = {}
cur = 0
total = len(data)
cont = True

def getHopCount(url):
    global cur
    global total
    global cont 
    global mp
    global mmp

    furl = tldextract.extract(url).fqdn
    try:
        if furl not in mmp:
            result, _ = traceroute(furl, maxttl=64, verbose=0, timeout=30)
            mmp[furl] = (average([snd[IP].ttl for snd, _ in result[IP]]))
        mp[url] = mmp[furl]
    except KeyboardInterrupt:
        print("Pika")
        raise
    except:
        mp[url] = 64
        mmp[furl] = 64
    return 1


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
        fututreHopCounts = {executor.submit(getHopCount, url): url for url in data['url']}
        done, not_done = concurrent.futures.wait(fututreHopCounts, timeout=0)
        try:
            while not_done:
                freshly_done, not_done = concurrent.futures.wait(not_done, timeout=1)
                done |= freshly_done
                for future in freshly_done:
                    cur += future.result()
                print(f"{cur}/{total}", end='\r', flush=True)
        except KeyboardInterrupt:
            print("Cancelling")
            for future in not_done:
                future.cancel()
            _ = concurrent.futures.wait(not_done, timeout=None)
            print("Cancelled")
        finally:
            data['hopCount'] = data['url'].map(mp)
            data.to_csv("tmp1.csv", index=False)
            print(data)
