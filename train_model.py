import os
import cv2
import face_recognition
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("Starting model training process...")

# --- Configuration ---
DATASET_PATH = 'dataset'
MODEL_FILE = "face_model.pkl"
SCALER_FILE = "scaler.pkl"
EMBEDDINGS_FILE = "face_embeddings.npy"
LABELS_FILE = "labels.npy"

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

# --- Data Loading and Processing ---
known_embeddings = []
known_labels = []
face_count_per_person = {}

print(f"Scanning dataset directory: '{DATASET_PATH}'")

# Check if dataset directory exists
if not os.path.exists(DATASET_PATH):
    print(f"\nError: Dataset directory '{DATASET_PATH}' does not exist!")
    print("Please create the directory and add subfolders for each person with their photos.")
    exit()

# Loop through each person in the dataset directory
for person_name in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person_name)
    
    # Skip any files that are not directories
    if not os.path.isdir(person_path):
        print(f"   - Skipping non-directory: {person_name}")
        continue

    print(f"\n-> Processing images for: {person_name}")
    face_count_per_person[person_name] = 0
    
    # Get all image files
    image_files = [f for f in os.listdir(person_path) 
                   if os.path.splitext(f.lower())[1] in SUPPORTED_FORMATS]
    
    if not image_files:
        print(f"   - Warning: No supported image files found for {person_name}")
        continue
    
    print(f"   - Found {len(image_files)} image files")
    
    # Loop through each image of the person
    for image_name in image_files:
        image_path = os.path.join(person_path, image_name)
        
        try:
            print(f"   - Processing: {image_name}")
            
            # Load the image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                print(f"     Warning: Could not read image {image_path}. Skipping.")
                continue
            
            # Convert BGR to RGB (face_recognition expects RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect face locations in the image
            # Using 'hog' model for speed, 'cnn' for accuracy (requires GPU)
            boxes = face_recognition.face_locations(rgb_image, model='hog')
            
            if not boxes:
                print(f"     Warning: No face detected in {image_name}")
                continue
            
            if len(boxes) > 1:
                print(f"     Warning: Multiple faces detected in {image_name}, using the first one")
            
            # Compute face embeddings for the detected faces
            embeddings = face_recognition.face_encodings(rgb_image, boxes)
            
            # If an embedding was found, add it and the corresponding label to our lists
            if embeddings:
                known_embeddings.append(embeddings[0])  # Take the first embedding
                known_labels.append(person_name)
                face_count_per_person[person_name] += 1
                print(f"     ✓ Face embedding extracted successfully")
            else:
                print(f"     Warning: Could not extract face encoding from {image_name}")

        except Exception as e:
            print(f"     Error processing {image_path}: {e}")

# Check if we have enough data
if not known_embeddings:
    print("\n❌ Error: No face embeddings were generated!")
    print("\nTroubleshooting tips:")
    print("1. Ensure your 'dataset' folder contains subfolders for each person")
    print("2. Each subfolder should contain clear photos of that person's face")
    print("3. Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif")
    print("4. Make sure faces are clearly visible and well-lit")
    exit()

print(f"\n📊 Training Data Summary:")
print(f"Total face embeddings: {len(known_embeddings)}")
print("Faces per person:")
for person, count in face_count_per_person.items():
    if count > 0:
        print(f"  - {person}: {count} faces")

# Warn if any person has too few samples
min_samples_per_person = 3
for person, count in face_count_per_person.items():
    if count > 0 and count < min_samples_per_person:
        print(f"⚠️  Warning: {person} has only {count} face samples. Consider adding more photos for better accuracy.")

print(f"\n🔬 Training the recognition model...")

# --- Model Training ---

# 1. Convert embeddings to numpy array for consistency
known_embeddings = np.array(known_embeddings)

# 2. Encode the string labels (names) into numerical format
label_encoder = LabelEncoder()
numerical_labels = label_encoder.fit_transform(known_labels)

# 3. Standardize the face embeddings for the SVM model
scaler = StandardScaler()
scaled_embeddings = scaler.fit_transform(known_embeddings)

# 4. Split data for validation (if we have enough samples)
if len(known_embeddings) >= 10:
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_embeddings, numerical_labels, test_size=0.2, random_state=42, stratify=numerical_labels
    )
    print(f"Using train/test split: {len(X_train)} training, {len(X_test)} testing samples")
else:
    X_train, y_train = scaled_embeddings, numerical_labels
    X_test, y_test = None, None
    print(f"Using all {len(known_embeddings)} samples for training (too few for train/test split)")

# 5. Train the Support Vector Machine (SVM) classifier
# Using probability=True allows us to get confidence scores later
print("Training SVM classifier...")
svm_model = SVC(kernel='linear', probability=True, C=1.0)
svm_model.fit(X_train, y_train)

# 6. Evaluate model if we have test data
if X_test is not None:
    print("\n📈 Model Evaluation:")
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.2%}")
    
    print("\nDetailed Classification Report:")
    target_names = [label_encoder.classes_[i] for i in sorted(set(y_test))]
    print(classification_report(y_test, y_pred, target_names=target_names))

# --- Save the Artifacts ---
print("\n💾 Saving model and supporting files...")

try:
    # 1. Save the trained SVM model
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(svm_model, f)
    print(f"✓ Saved main model to {MODEL_FILE}")

    # 2. Save the scaler
    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)
    print(f"✓ Saved data scaler to {SCALER_FILE}")

    # 3. Save the raw embeddings (for fallback recognition)
    np.save(EMBEDDINGS_FILE, known_embeddings)
    print(f"✓ Saved raw face embeddings to {EMBEDDINGS_FILE}")

    # 4. Save the original string labels (for fallback recognition)
    np.save(LABELS_FILE, known_labels)
    print(f"✓ Saved student labels to {LABELS_FILE}")

    # 5. Save additional metadata
    metadata = {
        'label_encoder': label_encoder,
        'face_count_per_person': face_count_per_person,
        'total_samples': len(known_embeddings),
        'unique_persons': len(set(known_labels))
    }
    with open("model_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print(f"✓ Saved model metadata to model_metadata.pkl")

except Exception as e:
    print(f"❌ Error saving files: {e}")
    exit()

print(f"\n✅ Model training complete! Your system is ready.")
print(f"   - Trained on {len(known_embeddings)} face samples")
print(f"   - Recognizes {len(set(known_labels))} different people")
print(f"   - Model files saved successfully")
print("\n🚀 Next steps:")
print("1. Make sure your Flask server (app.py) is stopped")
print("2. Restart the Flask server to load the new model")
print("3. Test the face recognition system")

# Display final statistics
print(f"\n📋 Final Dataset Statistics:")
unique_labels = list(set(known_labels))
for label in sorted(unique_labels):
    count = known_labels.count(label)
    print(f"   - {label}: {count} training samples")