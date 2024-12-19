// Function to reset modal content
function resetModalContent() {
    const modalBody = document.querySelector('#utilizeModal .modal-body');
    modalBody.innerHTML = `
        <!-- Loader within the modal -->
        <div id="modalLoader" class="text-center mt-3" style="display: none;">
            <div class="spinner-border text-secondary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p>Processing, please wait...</p>
        </div>

        <!-- Function Selection Section -->
        <div id="functionSelection">
            <label for="functionSelect">Select Function:</label>
            <select id="functionSelect" class="form-select mb-3">
                <option value="transcription">Transcription</option>
                <option value="sentiment-analysis">Sentiment Analysis</option>
                <option value="speech-to-text">Speech to Text</option>
            </select>
            <button id="applyFunctionBtn" class="btn">Apply Function</button>
        </div>
        <!-- Result Display Section -->
        <div id="functionResult" style="display: none; margin-top: 20px;">
            <!-- The transcription or other results will be displayed here -->
        </div>
    `;

    // Reattach event listener to the new "Apply Function" button
    document.getElementById('applyFunctionBtn').addEventListener('click', applyFunctionHandler);
}

// Extract the handler into a separate function
async function applyFunctionHandler() {
    const selectedFunction = document.getElementById('functionSelect').value;
    const speakerIndex = document.getElementById('applyFunctionBtn').getAttribute('data-speaker');

    if (!selectedFunction || speakerIndex === null) {
        alert("Please select a function and ensure a speaker is selected.");
        return;
    }

    const audioFilename = speakerIndex; // It's already the correct filename

    // Show loader during processing
    const loader = document.getElementById('modalLoader');
    loader.style.display = 'block';

    try {
        // Send the function and speaker filename to the server
        const response = await fetch('/apply_function', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                audio_filename: audioFilename,
                action: selectedFunction
            })
        });

        if (!response.ok) {
            const error = await response.json();
            console.error("Error response:", error);
            alert("Error: " + (error.message || "Function application failed."));
            return;
        }

        const result = await response.json();

        // Handle the response data
        if (selectedFunction === 'transcription' && result.transcription) {
            // Display the transcription in the modal
            const modalBody = document.querySelector('#utilizeModal .modal-body');
            modalBody.innerHTML = `
                <h5>Transcription Result:</h5>
                <div style="max-height: 300px; overflow-y: auto; margin-top: 15px;">
                    <p>${result.transcription}</p>
                </div>
                <button id="closeModalBtn" class="btn mt-3">Close</button>
            `;

            // Add event listener to the close button
            document.getElementById('closeModalBtn').addEventListener('click', function() {
                const modal = bootstrap.Modal.getInstance(document.getElementById('utilizeModal'));
                modal.hide();

                // Reset the modal content
                resetModalContent();
            });
        } else {
            // For other functions or if transcription is not available
            alert(`Function Applied: ${selectedFunction}\nResult: ${result.message}`);

            // Close the modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('utilizeModal'));
            modal.hide();
        }
    } catch (error) {
        console.error('Error:', error);
        alert("An error occurred while applying the function. Please try again.");
    } finally {
        loader.style.display = 'none'; // Hide loader after processing
    }
}

// Attach the handler initially
document.getElementById('applyFunctionBtn').addEventListener('click', applyFunctionHandler);

// Reset modal when it's hidden
const utilizeModal = document.getElementById('utilizeModal');

utilizeModal.addEventListener('hidden.bs.modal', function () {
    resetModalContent();
});

// Event listener for the "Utilize this Audio" button
document.getElementById('outputTable').addEventListener('click', function (event) {
    if (event.target && event.target.classList.contains('utilize-btn')) {
        const audioFilename = event.target.getAttribute('data-speaker');
        const modal = new bootstrap.Modal(document.getElementById('utilizeModal'));

        // Reset modal content
        resetModalContent();

        // Set the speaker filename to the modal button data attribute
        document.getElementById('applyFunctionBtn').setAttribute('data-speaker', audioFilename);

        // Show the modal
        modal.show();
    }
});

// Event listener for the file upload form
document.getElementById('uploadForm').addEventListener('submit', async function (event) {
    event.preventDefault();

    const fileInput = document.getElementById('audioFile');
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select a file!");
        return;
    }

    const formData = new FormData();
    formData.append('audio_file', file);

    // Show the loader
    const loader = document.getElementById('loader');
    loader.style.display = 'block';

    // Hide the table initially
    const table = document.getElementById('outputTable');
    table.style.display = 'none';

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            alert("Error: " + (error.message || "File upload failed."));
            loader.style.display = 'none'; // Hide the loader
            return;
        }

        const result = await response.json();

        if (result.urls && result.urls.length > 0) {
            const tbody = table.querySelector('tbody');
            tbody.innerHTML = '';

            result.urls.forEach((url, index) => {
                const speakerFilename = `${index}_merged.wav`; // Matches the actual filenames
                const row = `
                    <tr>
                        <td>Speaker ${index + 1}</td>
                        <td><a href="${url}" target="_blank" class="text-primary">View Output</a></td>
                        <td><button class="btn btn-secondary utilize-btn" data-speaker="${speakerFilename}">Utilize this Audio</button></td>
                    </tr>
                `;
                tbody.innerHTML += row;
            });

            table.style.display = 'table';
        } else {
            alert("No URLs were returned from the server.");
        }
    } catch (error) {
        console.error('Error:', error);
        alert("An error occurred. Please try again.");
    } finally {
        loader.style.display = 'none'; // Hide the loader after completion
    }
});