// ===================================================================================
// 1. GLOBAL VARIABLES & INITIALIZATION
// ===================================================================================

let stream = null;
let recognitionInterval = null;
const API_BASE_URL = 'http://127.0.0.1:5000/api';

/**
 * Main function that runs when the page is loaded.
 * It sets up global listeners and then runs page-specific code.
 */
document.addEventListener('DOMContentLoaded', function() {
    // This runs on EVERY page
    setActiveNavLink();
    setupGlobalEventListeners();

    // This is the "smart" part that runs code only for the current page
    if (document.getElementById('dashboard-section')) {
        loadDashboard();
    }
    if (document.getElementById('attendance-section')) {
        setupAttendancePage();
    }
    if (document.getElementById('students-section')) {
        setupStudentsPage();
    }
    if (document.getElementById('reports-section')) {
        setupReportsPage();
    }
});

// ===================================================================================
// 2. SETUP & EVENT LISTENER ATTACHMENT
// ===================================================================================

function setActiveNavLink() {
    const currentPage = window.location.pathname;
    const navLinks = document.querySelectorAll('a.nav-link');
    navLinks.forEach(link => {
        if (link.pathname === currentPage) {
            link.classList.add('bg-blue-700', 'font-semibold');
        }
    });
}

function setupGlobalEventListeners() {
    const addStudentBtn = document.getElementById('addStudentBtn');
    const addStudentModal = document.getElementById('addStudentModal');
    
    if (addStudentBtn) {
        addStudentBtn.addEventListener('click', showAddStudentModal);
    }

    if (addStudentModal) {
        const form = addStudentModal.querySelector('form');
        if (form) {
            form.addEventListener('submit', addStudent);
        }
        const closeButtons = addStudentModal.querySelectorAll('.close-modal-btn');
        closeButtons.forEach(btn => btn.addEventListener('click', hideAddStudentModal));
    }
}

function setupAttendancePage() {
    const subjects = ["Mathematics", "Physics", "Chemistry", "Biology", "Computer Science", "English"];
    const subjectSelect = document.getElementById('subjectSelect');
    if (subjectSelect) {
        subjectSelect.innerHTML = subjects.map(s => `<option value="${s}">${s}</option>`).join('');
    }
    
    document.getElementById('startCamera')?.addEventListener('click', startCamera);
    document.getElementById('stopCamera')?.addEventListener('click', stopCamera);
    document.getElementById('startRealtime')?.addEventListener('click', startRealtimeRecognition);
    document.getElementById('capturePhoto')?.addEventListener('click', captureAndRecognize);
}

function setupStudentsPage() {
    loadStudents();
    document.getElementById('studentSearch')?.addEventListener('input', filterStudents);
}

function setupReportsPage() {
    const subjects = ["", "Mathematics", "Physics", "Chemistry", "Biology", "Computer Science", "English"];
    const reportSubject = document.getElementById('reportSubject');
    if (reportSubject) {
        reportSubject.innerHTML = subjects.map(s => `<option value="${s}">${s || 'All Subjects'}</option>`).join('');
    }
    
    document.getElementById('reportStudent')?.addEventListener('input', debounce(generateReport, 500));
    document.getElementById('reportDate')?.addEventListener('change', generateReport);
    document.getElementById('reportSubject')?.addEventListener('change', generateReport);
    generateReport();
}

// ===================================================================================
// 3. PAGE-SPECIFIC & CORE FUNCTIONS
// ===================================================================================

// --- Dashboard ---
async function loadDashboard() {
    showLoading(true);
    try {
        console.log('Loading dashboard data...');
        
        const studentsResponse = await fetch(`${API_BASE_URL}/students`);
        console.log('Students response status:', studentsResponse.status);
        
        if (!studentsResponse.ok) {
            throw new Error(`Students API error: ${studentsResponse.status} ${studentsResponse.statusText}`);
        }
        
        const students = await studentsResponse.json();
        console.log('Students loaded:', students.length);
        
        const attendanceResponse = await fetch(`${API_BASE_URL}/attendance`);
        console.log('Attendance response status:', attendanceResponse.status);
        
        if (!attendanceResponse.ok) {
            throw new Error(`Attendance API error: ${attendanceResponse.status} ${attendanceResponse.statusText}`);
        }
        
        const attendance = await attendanceResponse.json();
        console.log('Attendance records loaded:', attendance.length);
        
        updateDashboardStats(students, attendance);
        updateRecentActivity(attendance);
        
        showNotification('Dashboard loaded successfully', 'success');
        
    } catch (error) {
        console.error('Error loading dashboard:', error);
        showNotification(`Error loading dashboard: ${error.message}`, 'error');
        
        // Show fallback data
        updateDashboardStats([], []);
        updateRecentActivity([]);
    } finally {
        showLoading(false);
    }
}

function updateDashboardStats(students, attendance) {
    const today = new Date().toISOString().split('T')[0];
    const presentToday = new Set(
        attendance
        .filter(record => record[1] && record[1].startsWith(today) && record[4] === 'present')
        .map(record => record[5])
    );
    const totalStudents = students.length;
    const presentCount = presentToday.size;
    const absentCount = totalStudents - presentCount;
    const rate = totalStudents > 0 ? Math.round((presentCount / totalStudents) * 100) : 0;
    
    const totalStudentsEl = document.getElementById('totalStudents');
    const presentTodayEl = document.getElementById('presentToday');
    const absentTodayEl = document.getElementById('absentToday');
    const attendanceRateEl = document.getElementById('attendanceRate');
    
    if (totalStudentsEl) totalStudentsEl.textContent = totalStudents;
    if (presentTodayEl) presentTodayEl.textContent = presentCount;
    if (absentTodayEl) absentTodayEl.textContent = absentCount;
    if (attendanceRateEl) attendanceRateEl.textContent = `${rate}%`;
}

function updateRecentActivity(attendance) {
    const recentActivity = document.getElementById('recentActivity');
    if (!recentActivity) return;
    
    const recent = attendance.slice(-5).reverse();
    if (recent.length > 0) {
        recentActivity.innerHTML = recent.map(record => `
            <div class="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div class="flex items-center">
                    <div class="w-2 h-2 rounded-full ${record[4] === 'present' ? 'bg-green-500' : 'bg-red-500'} mr-3"></div>
                    <div><p class="text-sm font-medium text-gray-900">${record[5] || 'Unknown'}</p><p class="text-xs text-gray-500">${record[3] || 'General'} - ${formatDate(record[1])}</p></div>
                </div>
                <span class="text-xs text-gray-500">${formatTime(record[1])}</span>
            </div>`).join('');
    } else {
        recentActivity.innerHTML = '<p class="text-gray-500 text-center">No recent activity</p>';
    }
}

// --- Attendance ---
async function startCamera() {
    if (stream) return;
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 640 }, 
                height: { ideal: 480 },
                facingMode: 'user'
            } 
        });
        const video = document.getElementById('video');
        const cameraOverlay = document.getElementById('cameraOverlay');
        const startCameraBtn = document.getElementById('startCamera');
        const stopCameraBtn = document.getElementById('stopCamera');
        const startRealtimeBtn = document.getElementById('startRealtime');
        const captureBtn = document.getElementById('capturePhoto');
        
        if (video) {
            video.srcObject = stream;
            video.play(); // Ensure video starts playing
        }
        if (cameraOverlay) cameraOverlay.style.display = 'none';
        if (startCameraBtn) startCameraBtn.classList.add('hidden');
        if (stopCameraBtn) stopCameraBtn.classList.remove('hidden');
        if (startRealtimeBtn) startRealtimeBtn.classList.remove('hidden');
        if (captureBtn) captureBtn.classList.remove('hidden');
        
        showNotification('Camera started successfully!', 'success');
    } catch (error) {
        console.error('Error starting camera:', error);
        showNotification('Could not start camera. Please check permissions and try again.', 'error');
    }
}

function stopCamera() {
    if (!stream) return;
    
    // Stop real-time recognition if running
    if (recognitionInterval) {
        clearInterval(recognitionInterval);
        recognitionInterval = null;
        const btn = document.getElementById('startRealtime');
        if (btn) {
            btn.innerHTML = '<i class="fas fa-sync-alt mr-2"></i>Start Real-time';
            btn.classList.replace('bg-red-500', 'bg-green-600');
        }
    }
    
    // Stop camera stream
    stream.getTracks().forEach(track => track.stop());
    stream = null;
    
    // Update UI
    const video = document.getElementById('video');
    const cameraOverlay = document.getElementById('cameraOverlay');
    const startCameraBtn = document.getElementById('startCamera');
    const stopCameraBtn = document.getElementById('stopCamera');
    const startRealtimeBtn = document.getElementById('startRealtime');
    const captureBtn = document.getElementById('capturePhoto');
    
    if (video) video.srcObject = null;
    if (cameraOverlay) cameraOverlay.style.display = 'flex';
    if (startCameraBtn) startCameraBtn.classList.remove('hidden');
    if (stopCameraBtn) stopCameraBtn.classList.add('hidden');
    if (startRealtimeBtn) startRealtimeBtn.classList.add('hidden');
    if (captureBtn) captureBtn.classList.add('hidden');
    
    // Clear face detection overlays
    clearFaceDetectionOverlay();
    
    showNotification('Camera stopped.', 'info');
}

function startRealtimeRecognition() {
    const button = document.getElementById('startRealtime');
    if (!button) return;
    
    if (recognitionInterval) {
        clearInterval(recognitionInterval);
        recognitionInterval = null;
        button.innerHTML = '<i class="fas fa-sync-alt mr-2"></i>Start Real-time';
        button.classList.replace('bg-red-500', 'bg-green-600');
        showNotification('Real-time recognition stopped.', 'info');
        clearFaceDetectionOverlay();
    } else {
        captureAndRecognize();
        recognitionInterval = setInterval(captureAndRecognize, 3000); // Every 3 seconds
        button.innerHTML = '<i class="fas fa-stop mr-2"></i>Stop Real-time';
        button.classList.replace('bg-green-600', 'bg-red-500');
        showNotification('Real-time recognition started.', 'success');
    }
}

async function captureAndRecognize() {
    if (!stream) {
        console.warn('No camera stream available');
        return;
    }
    
    const canvas = document.getElementById('canvas');
    const video = document.getElementById('video');
    
    if (!canvas || !video) {
        console.warn('Canvas or video element not found');
        return;
    }
    
    // Set canvas size to match video
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert canvas to blob and send for recognition
    canvas.toBlob(blob => {
        if (blob) {
            recognizeFace(blob);
        }
    }, 'image/jpeg', 0.8);
}

async function recognizeFace(imageBlob) {
    console.log('Starting face recognition...');
    const formData = new FormData();
    formData.append('image', imageBlob, 'capture.jpg');
    
    const subjectSelect = document.getElementById('subjectSelect');
    const subject = subjectSelect ? subjectSelect.value : 'General';
    formData.append('subject', subject);
    
    console.log('Sending recognition request for subject:', subject);
    
    try {
        showLoading(true);
        
        const response = await fetch(`${API_BASE_URL}/recognize`, { 
            method: 'POST', 
            body: formData 
        });
        
        console.log('Recognition response status:', response.status);
        console.log('Recognition response headers:', response.headers.get('content-type'));
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('Recognition error response:', errorText);
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        console.log('Recognition result:', result);
        
        // Check if the response has the expected structure
        if (!result.hasOwnProperty('recognized') || !result.hasOwnProperty('face_locations')) {
            console.error('Invalid response structure:', result);
            throw new Error('Invalid response from server');
        }
        
        // Display results
        displayRecognitionResults(result.recognized || []);
        
        // Draw face detection boxes
        if (result.face_locations && result.face_locations.length > 0) {
            console.log('Drawing face detection boxes:', result.face_locations.length, 'faces');
            drawFaceDetectionBoxes(result.face_locations);
        } else {
            console.log('No face locations to draw');
            clearFaceDetectionOverlay();
        }
        
        // Show notifications
        if (result.recognized && result.recognized.length > 0) {
            showNotification(`Recognized: ${result.recognized.join(', ')}`, 'success');
        } else if (result.total_faces > 0) {
            showNotification(`${result.total_faces} face(s) detected but not recognized`, 'warning');
        } else {
            showNotification('No faces detected in image', 'info');
        }
        
    } catch (error) {
        console.error('Error recognizing face:', error);
        showNotification(`Face recognition error: ${error.message}`, 'error');
        clearFaceDetectionOverlay();
    } finally {
        showLoading(false);
    }
}

function drawFaceDetectionBoxes(faceLocations) {
    const video = document.getElementById('video');
    if (!video) return;
    
    // Remove existing overlay
    clearFaceDetectionOverlay();
    
    // Create overlay div
    const overlay = document.createElement('div');
    overlay.id = 'face-detection-overlay';
    overlay.style.position = 'absolute';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.width = '100%';
    overlay.style.height = '100%';
    overlay.style.pointerEvents = 'none';
    overlay.style.zIndex = '10';
    
    // Get video container
    const videoContainer = video.parentElement;
    if (videoContainer) {
        videoContainer.style.position = 'relative';
        videoContainer.appendChild(overlay);
    }
    
    // Draw boxes for each detected face
    faceLocations.forEach(face => {
        const box = document.createElement('div');
        box.style.position = 'absolute';
        box.style.border = face.name === 'Unknown' ? '3px solid #ff4444' : '3px solid #44ff44';
        box.style.backgroundColor = 'transparent';
        box.style.borderRadius = '4px';
        
        // Calculate position relative to video element
        const videoRect = video.getBoundingClientRect();
        const scaleX = videoRect.width / video.videoWidth;
        const scaleY = videoRect.height / video.videoHeight;
        
        box.style.left = (face.left * scaleX) + 'px';
        box.style.top = (face.top * scaleY) + 'px';
        box.style.width = ((face.right - face.left) * scaleX) + 'px';
        box.style.height = ((face.bottom - face.top) * scaleY) + 'px';
        
        // Add label
        if (face.name) {
            const label = document.createElement('div');
            label.style.position = 'absolute';
            label.style.top = '-25px';
            label.style.left = '0';
            label.style.backgroundColor = face.name === 'Unknown' ? '#ff4444' : '#44ff44';
            label.style.color = 'white';
            label.style.padding = '2px 8px';
            label.style.fontSize = '12px';
            label.style.borderRadius = '3px';
            label.textContent = face.name;
            box.appendChild(label);
        }
        
        overlay.appendChild(box);
    });
}

function clearFaceDetectionOverlay() {
    const overlay = document.getElementById('face-detection-overlay');
    if (overlay) {
        overlay.remove();
    }
}

function displayRecognitionResults(recognized) {
    const resultsDiv = document.getElementById('recognitionResults');
    if (!resultsDiv) return;
    
    if (recognized.length > 0) {
        resultsDiv.innerHTML = `
            <div class="space-y-2">
                <p class="text-sm font-medium text-green-600">Recognized Students:</p>
                ${recognized.map(name => `
                    <div class="flex items-center p-2 bg-green-50 rounded border border-green-200">
                        <i class="fas fa-user-check text-green-500 mr-2"></i>
                        <span class="text-sm">${name}</span>
                    </div>
                `).join('')}
            </div>`;
    } else {
        resultsDiv.innerHTML = `
            <div class="text-center">
                <i class="fas fa-user-times text-red-500 text-2xl mb-2"></i>
                <p class="text-sm text-gray-600">No students recognized</p>
            </div>`;
    }
}

// --- Students ---
async function loadStudents() {
    showLoading(true);
    try {
        const response = await fetch(`${API_BASE_URL}/students`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        const students = await response.json();
        displayStudents(students);
    } catch (error) {
        console.error('Error loading students:', error);
        showNotification('Error loading students', 'error');
    } finally {
        showLoading(false);
    }
}

function displayStudents(students) {
    const tbody = document.getElementById('studentsTableBody');
    if (!tbody) return;
    
    if (students.length > 0) {
        tbody.innerHTML = students.map(student => `
            <tr class="hover:bg-gray-50">
                <td class="px-6 py-4 whitespace-nowrap">
                    <div class="flex items-center">
                        <div class="w-10 h-10 bg-blue-500 flex-shrink-0 rounded-full flex items-center justify-center text-white font-semibold">
                            ${student[1] ? student[1].charAt(0).toUpperCase() : '?'}
                        </div>
                        <div class="ml-4">
                            <div class="text-sm font-medium text-gray-900">${student[1] || 'N/A'}</div>
                        </div>
                    </div>
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${student[2] || 'N/A'}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${student[3] || 'N/A'}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${student[4] || 'N/A'}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm font-medium">
                    <button class="text-blue-600 hover:text-blue-700 mr-3">
                        <i class="fas fa-edit"></i>
                    </button>
                    <button class="text-red-600 hover:text-red-700">
                        <i class="fas fa-trash"></i>
                    </button>
                </td>
            </tr>`).join('');
    } else {
        tbody.innerHTML = `<tr><td colspan="5" class="text-center p-4 text-gray-500">No students found.</td></tr>`;
    }
}

function filterStudents() {
    const searchTerm = document.getElementById('studentSearch')?.value.toLowerCase() || '';
    const rows = document.querySelectorAll('#studentsTableBody tr');
    rows.forEach(row => {
        if (row.cells.length > 1) {
            const name = row.cells[0].textContent.toLowerCase();
            const roll = row.cells[1].textContent.toLowerCase();
            row.style.display = (name.includes(searchTerm) || roll.includes(searchTerm)) ? '' : 'none';
        }
    });
}

// --- Add Student ---
async function addStudent(event) {
    event.preventDefault();
    const form = event.target; 
    const formData = new FormData(form);

    // Validate required fields
    const name = formData.get('name');
    const rollNumber = formData.get('roll_number');
    const imageFile = formData.get('image');

    if (!name || !rollNumber || !imageFile || imageFile.size === 0) {
        showNotification('Name, Roll Number, and Image are required.', 'error');
        return;
    }

    showLoading(true);
    try {
        const response = await fetch(`${API_BASE_URL}/students`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.error || 'Failed to add student');
        }

        showNotification(result.message, 'success');
        hideAddStudentModal();
        
        // Refresh students list if on students page
        if (document.getElementById('students-section')) {
            loadStudents();
        }
        
        // Reset form
        form.reset();
        
    } catch (error) {
        console.error('Error adding student:', error);
        showNotification(error.message, 'error');
    } finally {
        showLoading(false);
    }
}

// --- Reports ---
async function generateReport() {
    const studentInput = document.getElementById('reportStudent');
    const dateInput = document.getElementById('reportDate');
    const subjectInput = document.getElementById('reportSubject');
    
    const student = studentInput ? studentInput.value : '';
    const date = dateInput ? dateInput.value : '';
    const subject = subjectInput ? subjectInput.value : '';
    
    const params = new URLSearchParams();
    if (student) params.append('student', student);
    if (date) params.append('date', date);
    if (subject) params.append('subject', subject);
    
    showLoading(true);
    try {
        const response = await fetch(`${API_BASE_URL}/attendance?${params}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        const records = await response.json();
        displayAttendanceRecords(records);
    } catch (error) {
        console.error('Error generating report:', error);
        showNotification('Error generating report', 'error');
    } finally {
        showLoading(false);
    }
}

function displayAttendanceRecords(records) {
    const tbody = document.getElementById('attendanceTableBody');
    if (!tbody) return;
    
    if (records.length > 0) {
        tbody.innerHTML = records.map(record => `
            <tr class="hover:bg-gray-50">
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${record[5] || 'Unknown'}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${formatDate(record[1])}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${record[3] || 'General'}</td>
                <td class="px-6 py-4 whitespace-nowrap">
                    <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${record[4] === 'present' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}">
                        ${record[4]}
                    </span>
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${record[2] || 'N/A'}</td>
            </tr>`).join('');
    } else {
        tbody.innerHTML = '<tr><td colspan="5" class="px-6 py-4 text-center text-gray-500">No records found for the selected filters.</td></tr>';
    }
}

// ===================================================================================
// 4. UTILITY & MODAL FUNCTIONS
// ===================================================================================

function showAddStudentModal() {
    const modal = document.getElementById('addStudentModal');
    if (modal) {
        modal.classList.remove('hidden');
    }
}

function hideAddStudentModal() {
    const modal = document.getElementById('addStudentModal');
    if (modal) {
        modal.classList.add('hidden');
        const form = modal.querySelector('form');
        if (form) form.reset();
    }
}

function showLoading(show) {
    const spinner = document.getElementById('loadingSpinner');
    if (spinner) {
        spinner.classList.toggle('hidden', !show);
    }
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    const colors = {
        success: 'bg-green-500', error: 'bg-red-500', 
        warning: 'bg-yellow-500', info: 'bg-blue-500'
    };
    notification.className = `fixed top-5 right-5 p-4 rounded-lg shadow-lg text-white z-50 transform transition-all duration-300 translate-x-full ${colors[type]}`;
    notification.innerHTML = `<span>${message}</span>`;
    document.body.appendChild(notification);
    setTimeout(() => notification.classList.remove('translate-x-full'), 100);
    setTimeout(() => {
        notification.classList.add('translate-x-full');
        setTimeout(() => notification.remove(), 300);
    }, 4000);
}

function formatDate(dateString) {
    return dateString ? new Date(dateString).toLocaleDateString() : 'N/A';
}

function formatTime(dateString) {
    return dateString ? new Date(dateString).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : 'N/A';
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
});