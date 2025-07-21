import oss2

access_key_id = 'your_access_key_id'
access_key_secret = 'your_access_key_secret'
auth = oss2.Auth(access_key_id, access_key_secret)

endpoint = "https://oss-cn-beijing.aliyuncs.com"
region = "cn-beijing"

bucket = oss2.Bucket(auth, endpoint, "hntcv", region=region)

def oss_upload(local_path, file_name):
    with open(local_path, 'rb') as fileobj:
        bucket.put_object(f'HNT_IMG/{file_name}', fileobj)
