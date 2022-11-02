// Select elemetns
const video = document.querySelector('video');
const picture = document.querySelector('.shot');

// Navigator video stream
async function videoStream (){
    try{
        const stream = await navigator.mediaDevices.getUserMedia({
            video:true,
            audio:false
        });
        // Set video source
        video.srcObject = stream;

        

    }catch(err){
        console.log(err);
    }
}
async function click(){
    //Take a picture on K press
    document.addEventListener('keypress', e =>{
        // console.log(e.code);
        if(e.code !== 'KeyK') return;
        // Create a Canvas
        const canvas = document.createElement('canvas');
        // Set canvas width and height
        canvas.width = video.width;
        canvas.height = video.height;
        // Draw a new image
        canvas.getContext('2d').drawImage(video, 0,0, video.width, video.height);
        // Take a shot
        let img = canvas.toDataURL('image/png').replace('image/png', 1.0);
        // Set Image src
        picture.src = img;
        // Save image file
        
        const anchorTag = document.createElement('a');
        anchorTag.href = img;
        anchorTag.download = 'my-image.png';
        document.body.appendChild(anchorTag);
        anchorTag.click();
    });
}

async function clickOn(){
    const canvas = document.createElement('canvas');
    canvas.width = video.width;
    canvas.height = video.height;
    canvas.getContext('2d').drawImage(video, 0,0, video.width, video.height);
    const img = canvas.toDataURL('image/png').replace('image/png', 1.0);
    picture.src = img;
    
    return img;
    // const anchorTag = document.createElement('a');
    //     anchorTag.href = img;
    //     anchorTag.download = 'my-image.png';
    //     document.body.appendChild(anchorTag);
    //     anchorTag.click();

}
// Run function
videoStream();