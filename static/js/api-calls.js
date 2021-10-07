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
    const higlightedSummaryText = document.getElementById("higlightedSummaryText")
    try {
        postText(e)
        .then(function (data) {
            if (data.success === true) {
                summaryText.innerText = data.summary
                higlightedSummaryText.innerHTML = data.higlighted_summary
            }
        })
    } catch (error) {
        console.log('error: ', error)        
        summaryText.innerText = 'Error: ' + error
        higlightedSummaryText.innerText = 'Error: ' + error
    }
}