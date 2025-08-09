# Automatic Student Attendance System

Flask-based face recognition attendance system using **Hybrid SVM + Cosine Similarity**. Features real-time recognition, MySQL integration, student management, and attendance analytics.

## 🌟 Features

- **Real-time Face Recognition** with live camera feed
- **Image Upload Support** for batch processing
- **Student Management** with CRUD operations
- **Attendance Analytics** and reporting
- **REST API** for frontend integration
- **MySQL Database** integration

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- MySQL Server
- Webcam/Camera

### Installation

1. **Clone and install**
```bash
git clone https://github.com/yourusername/Automatic_student_attendance_system.git
cd Automatic_student_attendance_system
pip install -r requirements.txt
```

2. **Setup Database**
```sql
CREATE DATABASE attendance_db;
```

3. **Configure MySQL** in `app.py`:
```python
app.config['MYSQL_HOST'] = '127.0.0.1'
app.config['MYSQL_PORT'] = 3307
app.config['MYSQL_USER'] = 'your_username'
app.config['MYSQL_PASSWORD'] = 'your_password'
```

4. **Prepare Training Data**
```
dataset/
├── student1_name/
│   ├── image1.jpg
│   └── image2.jpg
└── student2_name/
    └── image1.jpg
```

5. **Train Model & Run**
```bash
python train_model.py
python app.py
```

Visit `http://localhost:5000`

## 📖 Usage

- **Add Students**: Students page → Add New Student → Upload face images
- **Take Attendance**: Attendance page → Start Camera or Upload Image
- **View Reports**: Reports page → Filter by date/student

## 🔧 API Endpoints

```http
POST /api/recognize          # Face recognition
GET  /api/students           # Get students
POST /api/students           # Add student
GET  /api/attendance         # Get attendance records
GET  /api/health             # System health
```

## 🤖 Technical Details

- **Face Detection**: `face_recognition` library
- **ML Model**: SVM with cosine similarity validation
- **Database**: MySQL with students and attendance tables
- **Frontend**: HTML/CSS/JavaScript with Tailwind CSS

## 🔍 Troubleshooting

**Model files not found:** Run `python train_model.py`  
**MySQL connection failed:** Check server and credentials  
**Camera issues:** Verify permissions and OpenCV installation

## 📁 Project Structure

```
├── app.py              # Main Flask app
├── train_model.py      # ML training
├── requirements.txt    # Dependencies
├── static/js/          # Frontend
└── templates/          # HTML templates
```

---
**Built with ❤️ for automated attendance management**
