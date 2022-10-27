from labelbox import Client
from labelbox.data.annotation_types import Label, LabelList, ImageData, Point, ObjectAnnotation, Rectangle, Polygon
from labelbox.data.serialization import COCOConverter
import json
import os

if __name__ == '__main__':
    API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbDlxMXFlbDUwemkyMDd6eTQyN284cGh2Iiwib3JnYW5pemF0aW9uSWQiOiJjbDlwejRzY20waXplMDcxczlsdGoza3poIiwiYXBpS2V5SWQiOiJjbDlxMmRld2wyemRkMDcwemF5anphbzJqIiwic2VjcmV0IjoiYWFhNGJmZDk2NDYwZTYzZjNmMGMxYzczNDU4MDc4ZmYiLCJpYXQiOjE2NjY4MTQ1OTIsImV4cCI6MjI5Nzk2NjU5Mn0.ZkokMSHMVxZTzLb9w-RiVwm374l8rPVMgwgsdoyBDtc"
    client = Client(API_KEY)

    # dataset = client.get_dataset("cl9q1w5c7106y082bhhe6abs3")
    project = client.get_project("cl9q1c3i10pk8080c813qfaqy")

    labels = project.label_generator()
    clean_labels = []
    while(True):
        try:
            label = next(labels)
            clean_labels.append(label)
        except TypeError:
            print("Cant get data type")
        except StopIteration:
            print("Reached end of labels")
            break
    image_path = './weedimages/data/'

    coco_labels = COCOConverter.serialize_instances(
        clean_labels,
        image_root=image_path,
        ignore_existing_data=True
    )
    abs_path = os.path.abspath(str(coco_labels["info"]["image_root"]))
    coco_labels["info"]["image_root"] = abs_path
    # print(coco_labels)
    with open("weedimages/labels.json", "w") as outfile:
        json.dump(coco_labels, outfile)