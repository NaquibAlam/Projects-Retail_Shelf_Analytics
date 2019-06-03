            // ************************ Drag and drop ***************** //
            let dropArea = document.getElementById("drop-area");

            // Prevent default drag behaviors
            ;['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
              dropArea.addEventListener(eventName, preventDefaults, false);   
              document.body.addEventListener(eventName, preventDefaults, false);
            })

            // Highlight drop area when item is dragged over it
            ;['dragenter', 'dragover'].forEach(eventName => {
              dropArea.addEventListener(eventName, highlight, false);
            })

            ;['dragleave', 'drop'].forEach(eventName => {
              dropArea.addEventListener(eventName, unhighlight, false);
            })

            // Handle dropped files
            dropArea.addEventListener('drop', handleDrop, false);

            function preventDefaults (e) {
              e.preventDefault();
              e.stopPropagation();
            }

            function highlight(e) {
              dropArea.classList.add('highlight');
            }

            function unhighlight(e) {
              dropArea.classList.remove('highlight');
            }

            function handleDrop(e) {
              var dt = e.dataTransfer;
              var files = dt.files;

              handleFiles(files);
            }

            let uploadProgress = [];
            let progressBar = document.getElementById('progress-bar');

            function initializeProgress(numFiles) {
              // progressBar.value = 0;
              progressBar.style.width = "0%"
              uploadProgress = [];

              for(let i = numFiles; i > 0; i--) {
                uploadProgress.push(0);
              }
              document.getElementById('progress-container').style.display = "block";
            }

            function initializePreview() {
              let gallery = document.getElementById('gallery')
              while (gallery.hasChildNodes()) {
                gallery.removeChild(gallery.lastChild);
                }
            }

            function updateProgress(fileNumber, percent) {
              uploadProgress[fileNumber] = percent;
              let total = uploadProgress.reduce((tot, curr) => tot + curr, 0) / uploadProgress.length;
              console.debug('update', fileNumber, percent, total);
              // progressBar.value = total;
              progressBar.style.width = total+"%"
            }

            function uuidv4() {
                return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
                    var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
                    return v.toString(16);
                });
            }

            function handleFiles(files) {
              console.log('enter');
              files = [...files];
              if (files.length == 0) {
                // initialStates();
                return
              }
              initializeProgress(files.length);
              initializePreview();
              uuid = uuidv4();
              for (ind in files) {
                uploadFile(files[ind], ind, uuid);
              }
              // files.forEach(uploadFile);
              document.getElementById("label-box").style.display = "none";
              files.forEach(previewFile);
              document.getElementById('btn-circle-1').style.display = "block";
              document.getElementById('btn-circle-2').style.display = "block";
            }

            function previewFile(file) {
              let reader = new FileReader();
              reader.readAsDataURL(file);
              reader.onloadend = function() {
                let img = document.createElement('img');
                img.src = reader.result;
                document.getElementById('gallery').appendChild(img);
              }
            }

            function uploadFile(file, i, uuid) {
                // var form_data = new FormData($('#upload-file')[0]);
                // $.ajax({
                //     type: 'POST',
                //     url: '/uploadFiles',
                //     data: form_data,
                //     contentType: false,
                //     cache: false,
                //     processData: false,
                //     async: true,
                //     success: function(data) {
                //         console.log('Success!');
                //     },
                // });
              var url = '/uploadFiles'
              var xhr = new XMLHttpRequest()
              var formData = new FormData()
              xhr.open('POST', url, true)
              xhr.setRequestHeader('X-Requested-With', 'XMLHttpRequest')

              // Update progress (can be used to show progress indicator)
              xhr.upload.addEventListener("progress", function(e) {
                updateProgress(i, (e.loaded * 100.0 / e.total) || 100)
              })

              xhr.addEventListener('readystatechange', function(e) {
                if (xhr.readyState == 4 && xhr.status == 200) {
                  updateProgress(i, 100) // <- Add this
                }
                else if (xhr.readyState == 4 && xhr.status != 200) {
                  // Error. Inform the user
                }
              })

              formData.append('upload_preset', 'ujpu6gyk')
              formData.append('file[]', file)
              formData.append('file', uuid.toString())
              xhr.send(formData)
            }

            function initialStates() {
                document.getElementById('progress-container').style.display = "none";
                let gallery = document.getElementById('gallery')
                while (gallery.hasChildNodes()) {
                    gallery.removeChild(gallery.lastChild);
                }
                document.getElementById("label-box").style.display = "block";
                document.getElementById("btn-circle-1").style.display = "none";
                document.getElementById("btn-circle-2").style.display = "none";
            }

            function removeAll() {
                var form = {}
                form['uuid'] = uuid
                $.ajax({
                    type: 'POST',
                    url: '/dumpInput',
                    data: form,
                    contentType: 'application/x-www-form-urlencoded',
                    async: true,
                    success: function(data) {
                        console.log('Success!');
                    },
                });
                initialStates()
                // window.location.href = '/fileUpload'
            }

            function activateStitch() {
              var form = {}
              form['uuid'] = uuid
              $.ajax({
                type: 'POST',
                url: '/stitchAction',
                data: form,
                contentType: 'application/x-www-form-urlencoded',
                async: true,
                success: function(data) {
                  console.log('Success!');
                },
              });
              initialStates()
            }

            function activateCalc() {
              var form = {}
              form['uuid'] = uuid
              $.ajax({
                type: 'POST',
                url: '/calcAction',
                data: form,
                contentType: 'application/x-www-form-urlencoded',
                async: true,
                success: function(data) {
                  console.log('Success!');
                },
              });
              initialStates()
            }

            function start_long_task() {
              // add task status elements
              div = $('<div class="progress"><div></div><div>0%</div><div>...</div><div>&nbsp;</div></div><hr>');
              $('#progress').append(div);
              // create a progress bar
              var nanobar = new Nanobar({
                  bg: '#44f',
                  target: div[0].childNodes[0]
              });
              // send ajax POST request to start background job
              $.ajax({
                  type: 'POST',
                  url: '/longtask',
                  success: function(data, status, request) {
                      status_url = request.getResponseHeader('Location');
                      update_progress(status_url, nanobar, div[0]);
                  },
                  error: function() {
                      alert('Unexpected error');
                  }
              });
            }
            function update_progress(status_url, nanobar, status_div) {
                // send GET request to status URL
                $.getJSON(status_url, function(data) {
                    // update UI
                    percent = parseInt(data['current'] * 100 / data['total']);
                    nanobar.go(percent);
                    $(status_div.childNodes[1]).text(percent + '%');
                    $(status_div.childNodes[2]).text(data['status']);
                    if (data['state'] != 'PENDING' && data['state'] != 'PROGRESS') {
                        if ('result' in data) {
                            // show result
                            $(status_div.childNodes[3]).text('Result: ' + data['result']);
                        }
                        else {
                            // something unexpected happened
                            $(status_div.childNodes[3]).text('Result: ' + data['state']);
                        }
                    }
                    else {
                        // rerun in 2 seconds
                        setTimeout(function() {
                            update_progress(status_url, nanobar, status_div);
                        }, 2000);
                    }
                });
            }
            $(function() {
                $('#btn-circle-1').click(start_long_task);
            });