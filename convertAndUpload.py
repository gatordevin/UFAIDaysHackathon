from labelbox import Client


if __name__ == '__main__':
    API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbDlxMXFlbDUwemkyMDd6eTQyN284cGh2Iiwib3JnYW5pemF0aW9uSWQiOiJjbDlwejRzY20waXplMDcxczlsdGoza3poIiwiYXBpS2V5SWQiOiJjbDlxMmRld2wyemRkMDcwemF5anphbzJqIiwic2VjcmV0IjoiYWFhNGJmZDk2NDYwZTYzZjNmMGMxYzczNDU4MDc4ZmYiLCJpYXQiOjE2NjY4MTQ1OTIsImV4cCI6MjI5Nzk2NjU5Mn0.ZkokMSHMVxZTzLb9w-RiVwm374l8rPVMgwgsdoyBDtc"
    client = Client(API_KEY)

    dataset = client.get_dataset("cl9q1w5c7106y082bhhe6abs3")

    data_rows = dataset.data_rows()
    data_row = next(data_rows)
    print(data_row.labels())
    print("Associated dataset", data_row.dataset())
    print("Associated label(s)", next(data_row.labels()))
    print("External id", data_row.external_id)