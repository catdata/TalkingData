import csv

# ['event_id', 'app_id', 'is_installed', 'is_active']
csvfile = open('app_events.csv', newline='')
app_event = csv.reader(csvfile, delimiter=',', quotechar='|')
last_fine = app_event.__next__()
last_fine = app_event.__next__()

# ['event_id', 'device_id', 'timestamp', 'longitude', 'latitude']

csvfile2 = open('events.csv', newline='')
events = csv.reader(csvfile2, delimiter=',', quotechar='|')
events.__next__()

# ['app_id', 'label_id']
with open('app_labels.csv', newline='') as csvfile:
    app_labels = csv.reader(csvfile, delimiter=',', quotechar='|')
    apps = {}
    for app in app_labels:
        apps[app[0]] = app[1]

# ['label_id', 'category']
with open('label_categories.csv', newline='') as csvfile:
    label_categories = csv.reader(csvfile, delimiter=',', quotechar='|')
    labels = {}
    for label in label_categories:
        labels[label[0]] = label[1]

# ['device_id', 'phone_brand', 'device_model']
with open('phone_brand_device_model.csv', newline='') as csvfile:
    phone_brand_device_model = csv.reader(csvfile, delimiter=',', quotechar='|')
    devices = {}
    for device in phone_brand_device_model:
        devices[device[0]] = device[1:2]


with open('sortida.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

    for e in events:
        # event_id - installed_apps as app_id!category:* - active_apps as app_id!category:* - timestamp - longitude - latitude - phone_brand - device_model -device_id
        installed_apps = []
        active_apps = []

        # app_events last event
        while int(last_fine[0]) < int(e[0]):
            print("MASSA PETIT %s %s" % (e[0], last_fine[0]))
            last_fine = app_event.__next__()

        while int(last_fine[0]) == int(e[0]):
            if last_fine[2]:
                installed_apps.append("{app_id}!{category}".format(
                    app_id=last_fine[1],
                    category=labels[apps[last_fine[1]]]))

            if last_fine[3]:
                active_apps.append("{app_id}!{category}".format(
                    app_id=last_fine[1],
                    category=labels[apps[last_fine[1]]]))
            last_fine = app_event.__next__()
        if e[1] in devices and len(devices[e[1]]) > 1:
            device_model = devices[e[1]][1]
            brand = devices[e[1]][0]
        elif e[1] in devices:
            device_model = None
            brand = devices[e[1]][0]
        else:
            device_model = None
            brand = None
        writer.writerow([e[0], ':'.join(installed_apps), ':'.join(active_apps), e[2], e[3], e[4], brand, device_model, e[1]])

csvfile.close()
csvfile2.close()
