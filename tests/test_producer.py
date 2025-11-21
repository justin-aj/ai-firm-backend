from google.cloud import storage
import os
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\ajinf\2025\fall\webdev\ai-firm-backend\tests\service-key.json"


def upload_string(bucket_name, blob_name, text):
    client = storage.Client()  # Uses GOOGLE_APPLICATION_CREDENTIALS
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.upload_from_string(text, content_type="text/plain")
    print(f"Uploaded {blob_name}")

upload_string("test-direct", "test.txt", "hello world")
