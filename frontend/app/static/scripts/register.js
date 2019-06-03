/*******************************************************************************
 * *************** Java scripts used for various actions
 * ******************************
 ******************************************************************************/
/*--------------------------------User Registration Screen-----------------------------------*/

function Reg_User() {
	// Extract the variables
	var vFirstName = $("#inputFirstName").val();
	var vLastName = $("#inputLastName").val();
	var vEmailID = $("#inputEmail1").val();
	var vPwd = $("#inputPassword1").val();
	var vRePwd = $("#inputConfirmPassword1").val();
    
	// If no password
	if ((!vPwd) || (!vRePwd)) {
		console.log('Please enter your password.');
		return;
	}

	// If password do not match
	if (vPwd != vRePwd) {
		console.log('Passwords do not match.');
		return;
	}

	// Check type of first and last name
	if ((vFirstName.match(/\d/)) || (vLastName.match(/\d/))) {
		console.log("Names should not contain numbers") ;
		return;
	}

	// Check whether emailid is in valid format
	var filter = /^([\w-\.]+@([\w-]+\.)+[\w-]{2,4})?$/;

    if (!(filter.test(vEmailID))) {
    	console.log("Invalid format for email id") ;
        return; }

    // Check passwords constraints
    function Password_Validator(password) {
    	if (!(password.length > 8))
    		return false ;
    	if (!(password.match(/[A-z]/)))
    		return false ;
    	if (!(password.match(/[A-Z]/)))
    		return false ;
    	if (!(password.match(/\d/)))
    		return false ;
    	if (!(password.match(/[@#$%^&+=]/)))
            return false;
    	return true ;
    } 

    if (!(Password_Validator(vPwd))) {
    	console.log("Password is not in the correct form") ;
    	return ;
    }

    
    // fill up form variable for passing to backend
	var form = {};
	form["FirstName"] = vFirstName.toString();
	form["LastName"] = vLastName.toString();
	form["EmailID"] = vEmailID.toString();
	form["Password"] = vPwd.toString();

    var settings = {
        type: "POST",
        url: "/register_user",
        data: form,
        contentType: "application/x-www-form-urlencoded"
    };

    $.ajax(settings).done(function(response) {
        if (response.status == 'Success')
            window.location.href = "/index";
        else
        	console.log("Failure") ;

});
	
}