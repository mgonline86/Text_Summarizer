async function postText(e) {
    e.preventDefault()
    const originalText = document.getElementById("originalText").value
    let data = {'text' : originalText}
    data = JSON.stringify(data)
    
    const response = await fetch('/convert',{
        method: "POST",
		headers: {
            "Content-Type": "application/json",
			"Accept": "application/json"
		},
		body: data,
	});
    
    return response.json(); // parses JSON response into native JavaScript objects
}

function handleConvertText(e) {
    const summaryText = document.getElementById("summaryText")
    const img_nav = document.getElementById("nav-tab")
    const img_top_2_bottom_map = document.getElementById("nav-home")
    const img_central_map = document.getElementById("nav-profile")
    // const higlightedSummaryText = document.getElementById("higlightedSummaryText")
    try {
        postText(e)
        .then(function (data) {
            if (data.success === true) {
                summaryText.innerText = data.summary
                img_nav.classList.remove("hidden");
                while(img_top_2_bottom_map.firstChild) {
                    img_top_2_bottom_map.removeChild(img_top_2_bottom_map.firstChild);
                }
                while(img_central_map.firstChild) {
                    img_central_map.removeChild(img_central_map.firstChild);
                }

                // create a new timestamp     
                var timestamp = new Date().getTime();     
                    
                img_top_2_bottom_map.innerHTML = `<img src="static/img/top_2_bottom_map.png?t=${timestamp}" alt="top_2_bottom_map"/>`
                img_central_map.innerHTML = `<img src="static/img/central_map.png?t=${timestamp}" alt="central_map"/>`
                // higlightedSummaryText.innerHTML = data.higlighted_summary
            }
        })
    } catch (error) {
        console.log('error: ', error)        
        summaryText.innerText = 'Error: ' + error
        higlightedSummaryText.innerText = 'Error: ' + error
    }
}