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

async function handleConvertText(e) {
    const summaryText = document.getElementById("summaryText")
    const img_nav = document.getElementById("nav-tab")
    const img_top_2_bottom_map = document.getElementById("nav-home")
    const img_central_map = document.getElementById("nav-profile")
    const summart_count = document.getElementById("summary-count")
    const originalText = document.getElementById("originalText").value
    let data = {'text' : originalText}
    data = JSON.stringify(data)
    
    try {
        const jsonData = await postText(e);
        if (jsonData.success === true) {
            summaryText.innerText = jsonData.summary;
            summart_count.innerHTML = ""
            summart_count.innerHTML = `عدد حروف التلخيص: <span class="value" style="color: green;">${jsonData.summary.length}</span>`
            summart_count.classList.remove("hidden");
            while(img_top_2_bottom_map.firstChild) {
                img_top_2_bottom_map.removeChild(img_top_2_bottom_map.firstChild);
            }
            while(img_central_map.firstChild) {
                img_central_map.removeChild(img_central_map.firstChild);
            }

            var img_a_res = await fetch('/mindmap-a',{
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                body: data,
            });

            var img_b_res = await fetch('/mindmap-b',{
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                body: data,
            });

            const imgA_Blob = await img_a_res.blob();
            const imgB_Blob = await img_b_res.blob();

            const imgA_Tag = blobToImgTag(imgA_Blob);
            const imgB_Tag = blobToImgTag(imgB_Blob);

            imgA_Tag.alt = "Top to Bottom";
            imgB_Tag.alt = "Central";

            // append it to page
            img_top_2_bottom_map.appendChild(imgA_Tag);
            img_central_map.appendChild(imgB_Tag);
                
            img_nav.classList.remove("hidden");
        }
    
    } catch (error) {
        console.log('error: ', error)        
        summaryText.innerText = 'Error: ' + error
    }
}

function blobToImgTag(blob) {
    /*** Convert Blob to Image Tag */

    const imageUrl = URL.createObjectURL(blob);

    // create an image
    var outputImg = document.createElement('img');
    outputImg.src = imageUrl;

    return outputImg
}
