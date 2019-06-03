var login_modal = document.getElementById('LoginModal');

// Get the button that opens the modal
var login_btn = document.getElementById("login_butt");

window.onclick = function(event) {
    if (event.target == login_modal) {
        login_modal.style.display = "none";
    }
}

// When the user clicks the button, open the modal 
login_btn.onclick = function() {
    login_modal.style.display = "block";
}