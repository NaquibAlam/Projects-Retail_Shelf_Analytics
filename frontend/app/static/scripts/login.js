/*******************************************************************************
 * *************** Java scripts used for various actions
 * ******************************
 ******************************************************************************/
/*--------------------------------Login Screen-----------------------------------*/

$(document).ready(function(){
    var vRevisit = 'False' ;
    
    $("#remember-me").change(function() {
		if (this.checked)
			vRevisit = 'True';
	});
	console.log("Revisit")
	
	$("#login_button").click(function(){
		console.log("EmailID")
    	
		var vEmailID = $("#inputEmail").val();
		var vPwd = $("#inputPassword").val();
	    console.log(vEmailID)
    	console.log(vPwd)
    

	    if ((!vPwd)) {
			console.log('Please enter your password.');
			return;
		}

		var form = {};
			
		form["EmailID"] = vEmailID.toString();
		form["Password"] = vPwd.toString();
		form['Revisit'] = vRevisit.toString();

		var settings = {
			type: "POST",
			url: "/login_user",
			data: form,
			contentType: "application/x-www-form-urlencoded"
		};


		$.ajax(settings).done(function(response) {
			
			if (response.status == "Success"){
				sessionStorage.setItem("first_name", response.first_name);
				sessionStorage.setItem("last_name", response.last_name);    
				window.location.href = "/index";
			}
			else {
				console.log("failure") ;
			}
		});
    }) ;
}) ;