import uuid
import hashlib
import ntplib
import datetime
import psutil
import json

def read_expiry_date_from_json(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return datetime.datetime.strptime(data['expiry_date'], "%Y-%m-%d")
    except Exception as e:
        print(f"讀取過期日期失敗: {e}")
        sys.exit()

def list_mac_addresses():
    mac_addresses = []
    for interface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == psutil.AF_LINK:
                mac_addresses.append((interface, addr.address))
    return mac_addresses

def get_hardware_id():
    hardware_id = []
    # Get the MAC address
    mac_list = list_mac_addresses()
    for iface, mac in mac_list:
        # Generate hardware ID (you can combine different hardware info)
        hardware_id.append(hashlib.sha256(mac.encode()).hexdigest())
        #print(hardware_id)
    return hardware_id

def get_network_time():
    try:
        client = ntplib.NTPClient()
        response = client.request('pool.ntp.org')
        client = ntplib.NTPClient()
        response = client.request('pool.ntp.org')
        return datetime.datetime.fromtimestamp(response.tx_time)
    except:
        # Unable to get network time, handle exception
        return None
