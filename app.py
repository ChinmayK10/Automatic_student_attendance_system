from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_mysqldb import MySQL
import cv2
import numpy as np
import pickle
from datetime import datetime
import os
from sklearn.metrics.pairwise import cosine_similarity
import face_recognition
import logging
import traceback
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Enable debug mode
app.config['DEBUG'] = True

# === MySQL Configuration ===
app.config['MYSQL_HOST'] = '127.0.0.1'  # Changed from localhost
app.config['MYSQL_PORT'] = 3307
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Chinmay10'
app.config['MYSQL_DB'] = 'attendance_db'

# Initialize MySQL
try:
    mysql = MySQL(app)
    logger.info("✓ MySQL connection initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize MySQL: {e}")
    mysql = None

# === Load ML Models with Enhanced Error Handling ===
def load_models():
    """Load all ML models and embeddings with comprehensive error handling"""
    print("\n🤖 Loading ML models...")
    
    try:
        # Check if files exist first
        required_files = ['face_model.pkl', 'scaler.pkl', 'face_embeddings.npy', 'labels.npy']
        missing_files = []
        
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            logger.error(f"❌ Missing model files: {missing_files}")
            logger.error("Please run 'python train_model.py' first to generate the models")
            return None, None, None, None
        
        # Load SVM model
        with open("face_model.pkl", "rb") as f:
            svm_model = pickle.load(f)
        logger.info("✓ SVM model loaded successfully")
        
        # Load scaler
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        logger.info("✓ Scaler loaded successfully")
        
        # Load face embeddings
        face_embeddings = np.load("face_embeddings.npy")
        logger.info(f"✓ Face embeddings loaded: {face_embeddings.shape}")
        
        # Load labels
        labels = np.load("labels.npy")
        logger.info(f"✓ Labels loaded: {len(labels)} samples")
        
        # Validate data consistency
        if len(face_embeddings) != len(labels):
            logger.warning(f"⚠️  Mismatch: {len(face_embeddings)} embeddings vs {len(labels)} labels")
        
        unique_labels = set(labels)
        logger.info(f"✓ Unique students in model: {len(unique_labels)}")
        logger.info(f"Students: {list(unique_labels)}")
        
        return svm_model, scaler, face_embeddings, labels
        
    except FileNotFoundError as e:
        logger.error(f"❌ Model files not found: {e}")
        logger.error("Please run 'python train_model.py' first to generate the models")
        return None, None, None, None
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        logger.error(traceback.format_exc())
        return None, None, None, None

# Load models on startup
svm_model, scaler, face_embeddings, labels = load_models()

# === Enhanced Helper Functions ===
def extract_face_embeddings_with_drawing(image):
    """Extract face embeddings and return image with bounding boxes drawn"""
    try:
        logger.info("🔍 Starting enhanced face detection with bounding boxes...")
        
        # Convert BGR to RGB for face_recognition
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        logger.info(f"✓ Image converted to RGB: {rgb.shape}")
        
        # Detect faces using face_recognition library (same as Streamlit)
        boxes = face_recognition.face_locations(rgb, model='hog', number_of_times_to_upsample=1)
        logger.info(f"✓ Face detection complete: {len(boxes)} faces found")
        
        # If no faces found, try with enhanced detection
        if not boxes:
            logger.warning("⚠️  No faces detected with default settings, trying enhanced detection...")
            boxes = face_recognition.face_locations(rgb, model='cnn', number_of_times_to_upsample=2)
            logger.info(f"✓ Enhanced detection: {len(boxes)} faces found")
        
        if not boxes:
            logger.warning("❌ No faces detected in image")
            return [], [], image
        
        # Extract encodings
        logger.info("🔍 Extracting face encodings...")
        encodings = face_recognition.face_encodings(rgb, boxes, num_jitters=1)
        logger.info(f"✓ Face encodings extracted: {len(encodings)} encodings")
        
        # Create a copy of the original image for drawing
        result_image = image.copy()
        
        return boxes, encodings, result_image
        
    except Exception as e:
        logger.error(f"❌ Error extracting face embeddings: {e}")
        logger.error(traceback.format_exc())
        return [], [], image

def recognize_faces_and_draw(image, boxes, encodings):
    """Recognize faces and draw bounding boxes with labels (same logic as Streamlit)"""
    try:
        recognized = []
        face_locations = []
        result_image = image.copy()
        
        for i, ((top, right, bottom, left), embedding) in enumerate(zip(boxes, encodings)):
            logger.info(f"🔍 Processing face {i+1}/{len(encodings)}:")
            
            # Validate embedding
            if len(embedding) != 128:
                logger.error(f"❌ Invalid embedding size: {len(embedding)} (expected 128)")
                continue
            
            # Scale the embedding
            scaled = scaler.transform([embedding])
            
            # Try SVM first (same as Streamlit logic)
            svm_pred = svm_model.predict(scaled)[0]
            svm_proba = svm_model.predict_proba(scaled)[0]
            svm_conf = max(svm_proba)
            
            logger.info(f"✓ SVM prediction: '{svm_pred}' (confidence: {svm_conf:.3f})")
            
            # Use fallback cosine similarity if confidence is low (same threshold as Streamlit)
            if svm_conf < 0.6:  # Same threshold as Streamlit
                logger.info(f"⚠️  Low SVM confidence ({svm_conf:.3f}), using cosine similarity...")
                
                sims = cosine_similarity([embedding], face_embeddings)[0]
                max_idx = np.argmax(sims)
                max_sim = sims[max_idx]
                cosine_pred = labels[max_idx]
                
                logger.info(f"✓ Cosine similarity: '{cosine_pred}' (similarity: {max_sim:.3f})")
                
                if max_sim >= 0.6:  # Same threshold as Streamlit
                    final_pred = cosine_pred
                    confidence = max_sim
                    logger.info(f"✓ Using cosine similarity result")
                else:
                    final_pred = "Unknown"
                    confidence = max_sim
                    logger.info(f"⚠️  Below threshold, marked as Unknown")
            else:
                final_pred = svm_pred
                confidence = svm_conf
                logger.info(f"✓ Using SVM prediction (high confidence)")
            
            # Draw bounding box and label (EXACTLY like Streamlit)
            # Green box for recognized faces, red for unknown
            color = (0, 255, 0) if final_pred != "Unknown" else (0, 0, 255)
            
            # Draw rectangle
            cv2.rectangle(result_image, (left, top), (right, bottom), color, 2)
            
            # Draw label with background
            label = f"{final_pred} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(result_image, (left, top - label_size[1] - 10), 
                         (left + label_size[0], top), color, -1)
            cv2.putText(result_image, label, (left, top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Store results
            face_locations.append({
                "top": int(top), 
                "right": int(right), 
                "bottom": int(bottom), 
                "left": int(left),
                "name": final_pred,
                "confidence": float(confidence)
            })
            
            if final_pred != "Unknown":
                recognized.append(final_pred)
            
            logger.info(f"✅ Final prediction: '{final_pred}' (confidence: {confidence:.3f})")
        
        return recognized, face_locations, result_image
        
    except Exception as e:
        logger.error(f"❌ Error in face recognition and drawing: {e}")
        logger.error(traceback.format_exc())
        return [], [], image

def get_database_connection():
    """Get database connection with error handling"""
    try:
        if mysql and hasattr(mysql, 'connection') and mysql.connection:
            return mysql.connection
        else:
            logger.error("❌ MySQL connection not available")
            return None
    except Exception as e:
        logger.error(f"❌ Database connection error: {e}")
        return None

def mark_attendance(student_id, subject, session_id=None):
    """Mark attendance for a student with improved error handling"""
    try:
        connection = get_database_connection()
        if not connection:
            logger.error("❌ No database connection for marking attendance")
            return False
            
        cur = connection.cursor()
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate session_id if not provided
        if session_id is None:
            session_id = now.strftime("%Y-%m-%d_%H")
        
        # Check if already marked for this session
        cur.execute("SELECT id FROM attendance WHERE student_id = %s AND session_id = %s", (student_id, session_id))
        existing = cur.fetchone()
        
        if existing:
            logger.info(f"ℹ️  Attendance already marked for student_id {student_id} in session {session_id}")
            cur.close()
            return False
        
        # Insert new attendance record
        cur.execute("""
            INSERT INTO attendance (student_id, timestamp, session_id, subject, status)
            VALUES (%s, %s, %s, %s, 'present')
        """, (student_id, timestamp, session_id, subject))
        connection.commit()
        cur.close()
        
        logger.info(f"✓ Attendance marked for student_id {student_id}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error marking attendance: {e}")
        logger.error(traceback.format_exc())
        return False

# === TEMPLATE RENDERING ROUTES ===
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/attendance")
def attendance():
    return render_template("attendance.html")

@app.route("/students")
def students():
    return render_template("students.html")

@app.route("/reports")
def reports():
    return render_template("reports.html")

# === API ROUTES ===
@app.route("/api/recognize", methods=["POST"])
def recognize():
    """Enhanced face recognition endpoint with bounding box visualization"""
    print("\n" + "="*60)
    print("🎯 STARTING ENHANCED FACE RECOGNITION REQUEST")
    print("="*60)
    
    try:
        # === STEP 1: Request Validation ===
        logger.info("📋 Step 1: Validating request...")
        
        if 'image' not in request.files:
            logger.error("❌ No image file in request")
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            logger.error("❌ Empty filename")
            return jsonify({"error": "No file selected"}), 400
            
        subject = request.form.get('subject', 'General')
        logger.info(f"✓ Request valid - Subject: {subject}, File: {file.filename}")
        
        # === STEP 2: Model Availability Check ===
        logger.info("🤖 Step 2: Checking model availability...")
        
        if svm_model is None:
            logger.error("❌ SVM model is None")
            return jsonify({
                "error": "SVM model not loaded. Please train the model first using 'python train_model.py'",
                "recognized": [],
                "face_locations": [],
                "total_faces": 0
            }), 500
        
        if scaler is None or face_embeddings is None or labels is None:
            logger.error("❌ Required models/data not loaded")
            return jsonify({"error": "Required models not loaded"}), 500
            
        logger.info(f"✓ All models loaded successfully")
        
        # === STEP 3: Image Processing ===
        logger.info("🖼️  Step 3: Processing image...")
        
        # Read image data
        image_data = file.read()
        logger.info(f"✓ Image data read: {len(image_data)} bytes")
        
        if len(image_data) == 0:
            logger.error("❌ Image data is empty")
            return jsonify({"error": "Empty image file"}), 400
        
        # Decode image
        npimg = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.error("❌ Failed to decode image")
            return jsonify({"error": "Invalid image format or corrupted image"}), 400

        logger.info(f"✓ Image decoded successfully: {frame.shape}")
        
        # === STEP 4: Enhanced Face Detection and Drawing ===
        logger.info("👤 Step 4: Detecting faces with bounding box drawing...")
        
        try:
            boxes, encodings, drawn_image = extract_face_embeddings_with_drawing(frame)
            
            if len(boxes) == 0:
                logger.warning("⚠️  No faces detected in image")
                # Encode image as base64 to send back
                _, buffer = cv2.imencode('.jpg', frame)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                return jsonify({
                    "recognized": [],
                    "face_locations": [],
                    "total_faces": 0,
                    "status": "success",
                    "message": "No faces detected in the image. Please ensure faces are clearly visible.",
                    "processed_image": f"data:image/jpeg;base64,{img_base64}"
                })
            
            logger.info(f"✓ Face processing complete: {len(boxes)} faces, {len(encodings)} encodings")
            
        except Exception as face_error:
            logger.error(f"❌ Face detection error: {face_error}")
            logger.error(traceback.format_exc())
            return jsonify({
                "error": f"Face detection failed: {str(face_error)}",
                "recognized": [],
                "face_locations": [],
                "total_faces": 0
            }), 500
        
        # === STEP 5: Face Recognition and Drawing ===
        logger.info(f"🔍 Step 5: Recognizing {len(encodings)} faces and drawing results...")
        
        recognized, face_locations, result_image = recognize_faces_and_draw(frame, boxes, encodings)
        
        # === STEP 6: Database Operations ===
        logger.info(f"📊 Step 6: Processing attendance for recognized students...")
        
        attendance_marked = []
        for student_name in set(recognized):  # Remove duplicates
            try:
                connection = get_database_connection()
                if connection:
                    cur = connection.cursor()
                    cur.execute("SELECT id, name FROM students WHERE name = %s", (student_name,))
                    student = cur.fetchone()
                    cur.close()
                    
                    if student:
                        student_id, student_name_db = student
                        if mark_attendance(student_id, subject):
                            attendance_marked.append(student_name_db)
                            logger.info(f"✅ Attendance marked for {student_name_db}")
                        else:
                            attendance_marked.append(f"{student_name_db} (already marked)")
                            logger.info(f"ℹ️  Attendance already marked for {student_name_db}")
                    else:
                        logger.warning(f"❌ Student '{student_name}' not found in database")
                else:
                    logger.error(f"❌ Database connection failed")
            except Exception as db_error:
                logger.error(f"❌ Database error: {db_error}")
        
        # === STEP 7: Prepare Response with Processed Image ===
        logger.info(f"📤 Step 7: Preparing response with processed image...")
        
        # Encode the result image with bounding boxes as base64
        _, buffer = cv2.imencode('.jpg', result_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        response = {
            "recognized": attendance_marked,
            "face_locations": face_locations,
            "total_faces": len(boxes),
            "status": "success",
            "processed_image": f"data:image/jpeg;base64,{img_base64}",
            "message": f"Processed {len(boxes)} faces, recognized {len(set(recognized))} students"
        }
        
        logger.info(f"✅ Response prepared:")
        logger.info(f"  - Recognized: {len(attendance_marked)} students")
        logger.info(f"  - Face locations: {len(face_locations)} faces")
        logger.info(f"  - Total faces: {response['total_faces']}")
        logger.info(f"  - Students: {attendance_marked}")
        
        print("="*60)
        print("✅ ENHANCED FACE RECOGNITION COMPLETED SUCCESSFULLY")
        print("="*60)
        
        return jsonify(response)

    except Exception as e:
        logger.error(f"\n💥 CRITICAL ERROR in recognize endpoint:")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Full traceback:")
        logger.error(traceback.format_exc())
        
        print("="*60)
        print("❌ ENHANCED FACE RECOGNITION FAILED")
        print("="*60)
        
        return jsonify({
            "error": f"Recognition failed: {str(e)}",
            "error_type": type(e).__name__,
            "recognized": [],
            "face_locations": [],
            "total_faces": 0,
            "debug_info": str(e) if app.debug else None
        }), 500

@app.route("/api/students", methods=["GET"])
def get_students():
    """Get all students with error handling"""
    try:
        connection = get_database_connection()
        if not connection:
            return jsonify({"error": "Database connection failed"}), 500
            
        cur = connection.cursor()
        cur.execute("SELECT * FROM students ORDER BY name")
        students = cur.fetchall()
        cur.close()
        
        logger.info(f"✓ Retrieved {len(students)} students")
        return jsonify(students)
        
    except Exception as e:
        logger.error(f"❌ Error getting students: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/students", methods=["POST"])
def add_student():
    """Add new student with comprehensive validation"""
    try:
        # Extract form data
        name = request.form.get("name", "").strip()
        roll_number = request.form.get("roll_number", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        image_file = request.files.get("image")

        logger.info(f"📝 Adding new student: {name} (Roll: {roll_number})")

        # Validation
        if not name:
            return jsonify({"error": "Student name is required"}), 400
        
        if not roll_number:
            return jsonify({"error": "Roll number is required"}), 400
            
        if not image_file or image_file.filename == '':
            return jsonify({"error": "Student image is required"}), 400

        # Validate image format
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        file_ext = os.path.splitext(image_file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({"error": "Invalid image format. Use JPG, PNG, or BMP"}), 400

        connection = get_database_connection()
        if not connection:
            return jsonify({"error": "Database connection failed"}), 500

        cur = connection.cursor()

        # Check if roll number already exists
        cur.execute("SELECT id FROM students WHERE roll_number = %s", (roll_number,))
        if cur.fetchone():
            cur.close()
            return jsonify({"error": "Roll number already exists"}), 400

        # Check if name already exists
        cur.execute("SELECT id FROM students WHERE name = %s", (name,))
        if cur.fetchone():
            cur.close()
            return jsonify({"error": "Student name already exists"}), 400

        # Save image
        image_filename = f"{roll_number}_{name.replace(' ', '_')}{file_ext}"
        image_dir = os.path.join("static", "student_images")
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, image_filename)
        
        image_file.save(image_path)
        logger.info(f"✓ Image saved: {image_path}")

        # Insert student into database
        cur.execute("""
            INSERT INTO students (name, roll_number, email, phone, image_path)
            VALUES (%s, %s, %s, %s, %s)
        """, (name, roll_number, email or None, phone or None, image_path))
        
        connection.commit()
        student_id = cur.lastrowid
        cur.close()

        logger.info(f"✅ Student added successfully: {name} (ID: {student_id})")
        
        return jsonify({
            "message": f"Student '{name}' added successfully. Please retrain the model to enable face recognition.",
            "student_id": student_id,
            "note": "Run 'python train_model.py' to include this student in face recognition."
        })

    except Exception as e:
        logger.error(f"❌ Error adding student: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Failed to add student: {str(e)}"}), 500

@app.route("/api/attendance", methods=["GET"])
def get_attendance():
    """Get attendance records with flexible filtering"""
    try:
        # Get query parameters
        student = request.args.get("student", "").strip()
        date = request.args.get("date", "").strip()
        subject = request.args.get("subject", "").strip()

        connection = get_database_connection()
        if not connection:
            return jsonify({"error": "Database connection failed"}), 500

        # Build query
        query = """
            SELECT a.id, a.timestamp, a.session_id, a.subject, a.status, 
                   s.name, s.roll_number 
            FROM attendance a 
            JOIN students s ON a.student_id = s.id 
            WHERE 1=1
        """
        params = []

        if student:
            query += " AND (s.name LIKE %s OR s.roll_number LIKE %s)"
            params.extend([f"%{student}%", f"%{student}%"])
            
        if date:
            query += " AND DATE(a.timestamp) = %s"
            params.append(date)
            
        if subject:
            query += " AND a.subject = %s"
            params.append(subject)

        query += " ORDER BY a.timestamp DESC LIMIT 1000"  # Limit for performance

        cur = connection.cursor()
        cur.execute(query, tuple(params))
        records = cur.fetchall()
        cur.close()
        
        logger.info(f"✓ Retrieved {len(records)} attendance records")
        return jsonify(records)
        
    except Exception as e:
        logger.error(f"❌ Error getting attendance: {e}")
        return jsonify({"error": str(e)}), 500

# === HEALTH CHECK & SYSTEM STATUS ===
@app.route("/api/health", methods=["GET"])
def health_check():
    """Comprehensive system health check endpoint"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # Check database
        try:
            connection = get_database_connection()
            if connection:
                cur = connection.cursor()
                cur.execute("SELECT COUNT(*) FROM students")
                student_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM attendance")
                attendance_count = cur.fetchone()[0]
                cur.close()
                health_status["components"]["database"] = {
                    "status": "healthy",
                    "students": student_count,
                    "attendance_records": attendance_count
                }
            else:
                health_status["components"]["database"] = {"status": "unhealthy", "error": "No connection"}
        except Exception as e:
            health_status["components"]["database"] = {"status": "unhealthy", "error": str(e)}
        
        # Check models
        if all(x is not None for x in [svm_model, scaler, face_embeddings, labels]):
            health_status["components"]["ml_models"] = {
                "status": "healthy",
                "trained_students": len(set(labels)) if labels is not None else 0,
                "training_samples": len(labels) if labels is not None else 0
            }
        else:
            missing = []
            if svm_model is None: missing.append("svm_model")
            if scaler is None: missing.append("scaler")
            if face_embeddings is None: missing.append("face_embeddings")
            if labels is None: missing.append("labels")
            health_status["components"]["ml_models"] = {
                "status": "unhealthy",
                "missing": missing
            }
        
        # Check face_recognition library
        try:
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            face_recognition.face_locations(test_img)
            health_status["components"]["face_recognition"] = {"status": "healthy"}
        except Exception as e:
            health_status["components"]["face_recognition"] = {"status": "unhealthy", "error": str(e)}
        
        # Overall status
        component_statuses = [comp["status"] for comp in health_status["components"].values()]
        if "unhealthy" in component_statuses:
            health_status["status"] = "degraded"
        
        return jsonify(health_status)
        
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route("/api/system-info", methods=["GET"])
def system_info():
    """Get detailed system information for debugging"""
    try:
        info = {
            "flask_debug": app.debug,
            "model_files_exist": {
                "face_model.pkl": os.path.exists("face_model.pkl"),
                "scaler.pkl": os.path.exists("scaler.pkl"),
                "face_embeddings.npy": os.path.exists("face_embeddings.npy"),
                "labels.npy": os.path.exists("labels.npy")
            },
            "models_loaded": {
                "svm_model": svm_model is not None,
                "scaler": scaler is not None,
                "face_embeddings": face_embeddings is not None,
                "labels": labels is not None
            }
        }
        
        if labels is not None:
            info["training_data"] = {
                "total_samples": len(labels),
                "unique_students": len(set(labels)),
                "students": list(set(labels))
            }
            
        return jsonify(info)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === ERROR HANDLERS ===
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"❌ Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Print startup information
    print("\n" + "="*60)
    print("🚀 Enhanced Face Recognition Attendance System")
    print("="*60)
    
    # Check system health
    print(f"🤖 Models loaded: {all(x is not None for x in [svm_model, scaler, face_embeddings, labels])}")
    print(f"🗄️  Database configured: {mysql is not None}")
    
    if labels is not None:
        unique_students = set(labels)
        print(f"👥 Trained students: {len(unique_students)}")
        print(f"📸 Training samples: {len(labels)}")
        print(f"🎓 Student list: {list(unique_students)}")
    else:
        print("⚠️  No training data found - please run train_model.py")
    
    print("="*60)
    print("🌐 Server starting at: http://127.0.0.1:5000")
    print("📋 Health check: http://127.0.0.1:5000/api/health")
    print("🔍 System info: http://127.0.0.1:5000/api/system-info")
    print("🐛 Debug mode: ENABLED")
    print("🎯 Enhanced Features:")
    print("   ✓ Real-time bounding box drawing")
    print("   ✓ Streamlit-compatible face detection")
    print("   ✓ Processed image return with annotations")
    print("   ✓ Hybrid SVM + Cosine similarity (same as Streamlit)")
    print("="*60)
    print("\n💡 The enhanced backend now matches your working Streamlit implementation!")
    print("   - Same face detection algorithm")
    print("   - Same recognition thresholds")
    print("   - Green boxes for recognized faces")
    print("   - Returns processed image with annotations")
    print("\n")
    
    # Start Flask application
    app.run(debug=True, host='0.0.0.0', port=5000)