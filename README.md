# Automatic Student Attendance System

A Flask-based **Face Recognition Attendance System** utilizing **Hybrid SVM + Cosine Similarity** for accurate student identification. This system provides real-time face recognition, comprehensive student management, attendance tracking, and detailed analytics through a modern web interface.

## 🌟 Features

### Core Functionality
- **Real-time Face Recognition**: Live camera feed for instant student identification
- **Hybrid ML Model**: Combines SVM classification with cosine similarity for improved accuracy
- **Image Upload Support**: Recognize students from uploaded images
- **Attendance Tracking**: Automatic attendance marking with timestamp logging
- **Student Management**: Complete CRUD operations for student records

### Advanced Features
- **MySQL Integration**: Robust database management for students and attendance records
- **Analytics Dashboard**: Comprehensive attendance reports and statistics
- **Search & Filter**: Advanced filtering options for attendance records
- **REST API**: Full API endpoints for frontend integration
- **Model Training**: Automated face encoding and SVM model training
- **Health Monitoring**: System health checks and diagnostics

### User Interface
- **Modern Web UI**: Responsive design with Tailwind CSS
- **Interactive Dashboard**: Real-time statistics and visualizations
- **Multi-page Application**: Dedicated pages for attendance, students, and reports
- **Real-time Updates**: Live camera feed with face detection overlays

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- MySQL Server (running on port 3307)
- Webcam/Camera for real-time recognition

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Automatic_student_attendance_system.git
cd Automatic_student_attendance_system
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install flask flask-cors flask-mysqldb opencv-python numpy scikit-learn face-recognition pillow dlib
```

3. **Setup MySQL Database**
```sql
CREATE DATABASE attendance_db;
USE attendance_db;

CREATE TABLE students (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    roll_number VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100),
    phone VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE attendance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    student_name VARCHAR(100) NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    confidence FLOAT,
    method VARCHAR(50)
);
```

4. **Configure Database Connection**
Update the MySQL configuration in `app.py`:
```python
app.config['MYSQL_HOST'] = '127.0.0.1'
app.config['MYSQL_PORT'] = 3307
app.config['MYSQL_USER'] = 'your_username'
app.config['MYSQL_PASSWORD'] = 'your_password'
app.config['MYSQL_DB'] = 'attendance_db'
```

5. **Prepare Training Data**
Create a `dataset` folder and organize student images:
```
dataset/
├── student1_name/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── student2_name/
│   ├── image1.jpg
│   └── ...
```

6. **Train the Model**
```bash
python train_model.py
```

7. **Run the Application**
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## 📖 Usage Guide

### Adding Students
1. Navigate to the **Students** page
2. Click **"Add New Student"**
3. Fill in student details (name, roll number, email, phone)
4. Upload multiple face images for better recognition accuracy
5. The system will automatically extract face encodings

### Recording Attendance
1. Go to the **Attendance** page
2. **Live Camera**: Click "Start Camera" for real-time recognition
3. **Image Upload**: Upload an image containing student faces
4. The system will identify students and mark attendance automatically

### Viewing Reports
1. Access the **Reports** page
2. Filter by date range, student name, or attendance status
3. Export attendance data in various formats
4. View analytics and attendance statistics

## 🔧 API Documentation

### Base URL
```
http://localhost:5000/api
```

### Endpoints

#### Face Recognition
```http
POST /api/recognize
Content-Type: application/json

{
    "image": "base64_encoded_image_string"
}
```

**Response:**
```json
{
    "success": true,
    "students_found": [
        {
            "name": "John Doe",
            "confidence": 0.95,
            "bbox": [x, y, w, h]
        }
    ],
    "attendance_marked": ["John Doe"],
    "processed_image": "base64_encoded_result_image"
}
```

#### Students Management
```http
GET /api/students
POST /api/students
```

#### Attendance Records
```http
GET /api/attendance?date=YYYY-MM-DD&student=name
```

#### System Health
```http
GET /api/health
GET /api/system-info
```

## 🤖 Machine Learning Architecture

### Model Components
- **Face Detection**: Using `face_recognition` library with HOG/CNN models
- **Feature Extraction**: 128-dimensional face encodings
- **Classification**: Support Vector Machine (SVM) with RBF kernel
- **Similarity Matching**: Cosine similarity for enhanced accuracy
- **Preprocessing**: StandardScaler for feature normalization

### Training Process
1. **Data Collection**: Extract faces from student images
2. **Encoding Generation**: Create 128D face embeddings
3. **Data Preprocessing**: Normalize features using StandardScaler
4. **Model Training**: Train SVM classifier on face encodings
5. **Model Persistence**: Save trained models and encodings

### Recognition Pipeline
1. **Face Detection**: Locate faces in input image
2. **Feature Extraction**: Generate face encodings
3. **Classification**: Predict using trained SVM model
4. **Verification**: Validate using cosine similarity
5. **Confidence Scoring**: Calculate recognition confidence

## 🗂️ Project Structure

```
Automatic_student_attendance_system/
├── app.py                 # Main Flask application
├── train_model.py         # ML model training script
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── face_model.pkl         # Trained SVM model
├── scaler.pkl            # Feature scaler
├── face_embeddings.npy   # Face encoding database
├── labels.npy            # Student labels
├── static/
│   └── js/
│       └── script.js     # Frontend JavaScript
├── templates/
│   ├── layout.html       # Base template
│   ├── index.html        # Dashboard
│   ├── attendance.html   # Attendance page
│   ├── students.html     # Student management
│   └── reports.html      # Reports page
└── dataset/              # Training images (create this)
    ├── student1/
    ├── student2/
    └── ...
```

## ⚙️ Configuration

### Environment Variables
```bash
FLASK_ENV=development
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3307
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DB=attendance_db
```

### Model Parameters
```python
# SVM Configuration
SVM_KERNEL = 'rbf'
SVM_C = 1.0
SVM_GAMMA = 'scale'

# Face Detection
FACE_MODEL = 'hog'  # or 'cnn' for better accuracy
UPSAMPLING = 1      # Increase for better detection
NUM_JITTERS = 1     # Face encoding consistency
```

## 🔍 Troubleshooting

### Common Issues

**1. Model files not found**
```bash
python train_model.py
```

**2. MySQL connection failed**
- Check MySQL server is running on port 3307
- Verify credentials in `app.py`
- Ensure database `attendance_db` exists

**3. Camera not working**
- Check camera permissions
- Verify OpenCV installation
- Try different camera indices

**4. Low recognition accuracy**
- Add more training images per student
- Ensure good lighting in training images
- Use high-quality, front-facing photos

### Debug Mode
Enable debug logging by setting:
```python
app.config['DEBUG'] = True
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **face_recognition** library by Adam Geitgey
- **OpenCV** for computer vision operations
- **scikit-learn** for machine learning algorithms
- **Flask** for web framework
- **MySQL** for database management

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Email: [your-email@domain.com]
- Documentation: [Wiki/Docs Link]

---
**Built with ❤️ for automated attendance management**
