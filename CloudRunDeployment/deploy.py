# deploy.pp

from google.cloud import aiplatform

PROJECT_ID = "heartpcg"   
LOCATION   = "us-central1"
IMAGE_URI  = f"us-central1-docker.pkg.dev/heartpcg/heart-pcg-repo/heart-pcg-api:v1"

aiplatform.init(project=PROJECT_ID, location=LOCATION)

# Register the model
print("Registering model...")
model = aiplatform.Model.upload(
    display_name="heart-pcg-vgg16-densenet121",
    serving_container_image_uri=IMAGE_URI,
    serving_container_ports=[8080],
    serving_container_predict_route="/predict",
    serving_container_health_route="/health",
)
print(f"Model registered: {model.resource_name}")

# Create endpoint
print("Creating endpoint...")
endpoint = aiplatform.Endpoint.create(
    display_name="heart-pcg-endpoint"
)
print(f"Endpoint created: {endpoint.resource_name}")

# Deploy model to endpoint
print("Deploying model (this takes ~10 minutes)...")
model.deploy(
    endpoint=endpoint,
    machine_type="n1-standard-4",
    min_replica_count=1,
    max_replica_count=2,
    traffic_split={"0": 100},
)

print("Deployment complete!")
print(f"Endpoint ID: {endpoint.name}")
print(f"Full resource name: {endpoint.resource_name}")
print("\nSave this endpoint ID — your mobile app will use it.")