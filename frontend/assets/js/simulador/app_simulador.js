// ===== TACHO SIMULADOR - App Logic =====

let currentStream = null;
let capturedBlob = null;
let verifiedStudent = null;

// DOM Elements
const video = document.getElementById('webcam-video');
const canvas = document.getElementById('webcam-canvas');
const btnCapture = document.getElementById('btn-capture');
const btnRetake = document.getElementById('btn-retake');
const btnSend = document.getElementById('btn-send');
const spinner = document.getElementById('classify-spinner');

// ===== ALERT SYSTEM =====
function showSimAlert(type, message) {
    const alert = document.getElementById('sim-alert');
    const icon = document.getElementById('sim-alert-icon');
    const msg = document.getElementById('sim-alert-message');
    alert.className = `alert alert-${type} show`;
    icon.textContent = type === 'success' ? '\u2713' : type === 'error' ? '\u2717' : '\u24D8';
    msg.textContent = message;
    setTimeout(() => hideSimAlert(), 5000);
}

function hideSimAlert() {
    const alert = document.getElementById('sim-alert');
    alert.classList.remove('show');
}

// ===== STEP 1: DNI VERIFICATION =====
async function verificarDNI(e) {
    e.preventDefault();
    const dni = document.getElementById('sim-dni').value.trim();

    if (!dni || dni.length < 6) {
        showSimAlert('error', 'Ingrese un DNI valido (minimo 6 caracteres)');
        return false;
    }

    try {
        const res = await fetch(`${CONFIG.API_BASE_URL}${CONFIG.ENDPOINTS.VERIFICAR_DNI}/${dni}`, {
            method: 'GET'
        });

        const data = await res.json();

        if (res.ok && data.usuario_id) {
            verifiedStudent = { ...data, dni: dni };
            document.getElementById('student-name').textContent = data.nombre;
            document.getElementById('student-aula').textContent = `Aula ID: ${data.aula_id || '--'}`;
            document.getElementById('dni-result').style.display = 'flex';
            showSimAlert('success', 'Estudiante verificado correctamente');
            
            // Open webcam after short delay
            setTimeout(() => {
                document.getElementById('step-webcam').classList.add('visible');
                startWebcam();
            }, 600);
        } else {
            showSimAlert('error', data.detail || 'DNI no encontrado en el sistema');
        }
    } catch (err) {
        showSimAlert('error', 'Error de conexion al verificar DNI');
        console.error('DNI verification error:', err);
    }

    return false;
}

// ===== STEP 2: WEBCAM =====
async function startWebcam() {
    try {
        const constraints = {
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'environment'
            }
        };
        currentStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = currentStream;
        video.style.display = 'block';
        canvas.style.display = 'none';
        btnCapture.style.display = 'block';
        btnRetake.style.display = 'none';
        btnSend.style.display = 'none';
    } catch (err) {
        showSimAlert('error', 'No se pudo acceder a la camara. Verifique permisos.');
        console.error('Webcam error:', err);
    }
}

function stopWebcam() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
}

function capturePhoto() {
    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    // Show canvas, hide video
    video.style.display = 'none';
    canvas.style.display = 'block';
    btnCapture.style.display = 'none';
    btnRetake.style.display = 'inline-block';
    btnSend.style.display = 'inline-flex';

    // Convert canvas to blob
    canvas.toBlob(blob => {
        capturedBlob = blob;
    }, 'image/jpeg', 0.85);
}

function retakePhoto() {
    video.style.display = 'block';
    canvas.style.display = 'none';
    btnCapture.style.display = 'block';
    btnRetake.style.display = 'none';
    btnSend.style.display = 'none';
    capturedBlob = null;
}

// ===== STEP 2 -> 3: SEND TO BACKEND =====
async function sendForClassification() {
    if (!verifiedStudent) {
        showSimAlert('error', 'Primero verifique el DNI del estudiante');
        return;
    }

    // Extract base64 from canvas (strip the data:image/jpeg;base64, prefix)
    const base64Data = canvas.toDataURL('image/jpeg', 0.85);
    const base64String = base64Data.split(',')[1];

    if (!base64String) {
        showSimAlert('error', 'Primero capture una foto del residuo');
        return;
    }

    // Stop webcam and show loading
    stopWebcam();
    spinner.classList.add('visible');
    btnSend.style.display = 'none';
    btnRetake.style.display = 'none';

    const formData = new FormData();
    formData.append('imagen_base64', base64String);
    formData.append('usuario_id', verifiedStudent.usuario_id);

    try {
        const res = await fetch(`${CONFIG.API_BASE_URL}${CONFIG.ENDPOINTS.CLASIFICAR_RESIDUO}`, {
            method: 'POST',
            body: formData
        });

        const data = await res.json();
        spinner.classList.remove('visible');

        if (res.ok && data.tipo_residuo_detectado) {
            showResult(data);
        } else {
            showSimAlert('error', data.detail || 'Error al clasificar el residuo');
            // Restart webcam so user can retry
            startWebcam();
            btnSend.style.display = 'none';
        }
    } catch (err) {
        spinner.classList.remove('visible');
        // Restart webcam so user can retry
        startWebcam();
        btnSend.style.display = 'none';
        showSimAlert('error', 'Error de conexion al clasificar. Verifique que el servidor este activo.');
        console.error('Classification error:', err);
    }
}

// ===== STEP 3: SHOW RESULT =====
async function showResult(data) {
    stopWebcam();

    // Hide webcam step
    document.getElementById('step-webcam').classList.remove('visible');

    // Show result
    const resultSection = document.getElementById('step-result');
    resultSection.classList.add('visible');

    const categoryMap = {
        'glass': 'Vidrio',
        'organic': 'Organico',
        'metal': 'Metal',
        'others': 'Otros',
        'plastic': 'Plastico',
        'paper': 'Papel'
    };

    const category = data.tipo_residuo_detectado || '--';
    const confidence = data.confianza || 0;
    const puntosGanados = data.puntos_sumados || 10;

    document.getElementById('result-category').textContent = categoryMap[category] || category;
    document.getElementById('result-confidence').textContent = `Confianza: ${(confidence * 100).toFixed(1)}% | +${puntosGanados} pts ganados`;

    // Fetch updated total score
    let totalScore = puntosGanados;
    try {
        const scoreRes = await fetch(`${CONFIG.API_BASE_URL}/api/dashboard/alumno/${verifiedStudent.usuario_id}`);
        if (scoreRes.ok) {
            const scoreData = await scoreRes.json();
            totalScore = scoreData.puntos_totales || puntosGanados;
        }
    } catch (e) { /* silent fallback */ }

    const resultAlert = document.getElementById('result-alert');
    resultAlert.className = 'alert alert-success show';
    resultAlert.innerHTML = `
        <span style="display:flex; flex-direction:column; align-items:center; gap:0.25rem;">
            <strong>Residuo "${categoryMap[category] || category}" detectado correctamente</strong>
            <span>Imagen procesada exitosamente para ${verifiedStudent.nombre}</span>
            <span style="font-weight:700; font-size:1.1rem; margin-top:0.25rem;">Puntaje total: ${totalScore} pts</span>
        </span>
    `;

    // Show prominent notification
    showSimAlert('success', `+${puntosGanados} pts! Material detectado: ${categoryMap[category] || category}`);

    // Show achievement notification if any
    if (data.nuevo_logro) {
        setTimeout(() => {
            showSimAlert('success', `Nuevo logro desbloqueado: ${data.nuevo_logro.nombre}`);
        }, 3000);
    }
}

// ===== RESET =====
function resetSimulator() {
    stopWebcam();
    capturedBlob = null;
    verifiedStudent = null;

    // Reset UI
    document.getElementById('sim-dni').value = '';
    document.getElementById('dni-result').style.display = 'none';
    document.getElementById('step-webcam').classList.remove('visible');
    document.getElementById('step-result').classList.remove('visible');
    spinner.classList.remove('visible');
    btnSend.style.display = 'none';

    hideSimAlert();
}

// ===== EVENT LISTENERS =====
btnCapture.addEventListener('click', capturePhoto);
btnRetake.addEventListener('click', retakePhoto);
btnSend.addEventListener('click', sendForClassification);
