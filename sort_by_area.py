import os

import labelimg_xml

xml_dir = os.path.join(os.getcwd(), 'xml')
file_list = [_ for _ in os.listdir(xml_dir) if _.endswith('.xml')]

for file in file_list:
    title, table = labelimg_xml.read_xml(xml_dir, '/' + file)
    for box in table:
        print(box, (box[3]-box[1])*(box[4]-box[2]))
    table.sort(key=lambda x: (x[3]-x[1])*(x[4]-x[2]))
    for box in table:
        print(box, (box[3]-box[1])*(box[4]-box[2]))

    labelimg_xml.write_xml(title, table, xml_dir + '/', file)