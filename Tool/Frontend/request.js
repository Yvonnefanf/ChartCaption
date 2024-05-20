let headers = new Headers();
headers.append('Content-Type', 'application/json');
headers.append('Accept', 'application/json');


function getAnswer(question) {
    fetch(`${window.location.href}get_answer`, {
        method: 'POST',
        body: JSON.stringify({
           'question':question
        }),
        headers: headers,
        mode: 'cors'
    })
    .then(response => response.json())
    .then(res => {
        console.log('res',res)
        window.vueApp.messages = res.msglist
        window.vueApp.isGeneratingAnswer = false
        window.vueApp.question = ''
    })
    .catch(error => {
        console.error('Error fetching data:', error);
        window.vueApp.isGeneratingAnswer = false
        window.vueApp.isCanvasLoading = false
        window.vueApp.$message({
            type: 'error',
            message: `Unknown Backend Error`
          });
    });
}

function getHistorymsg(){
    fetch(`${window.location.href}get_historymsg`,{
        method: 'GET',
        headers: headers,
        mode: 'cors'
    })
    .then(response => response.json())
    .then(res => {
        console.log('res',res)
        window.vueApp.messages = res.msglist
    })
    .catch(error => {
        console.error('Error fetching data:', error);
        window.vueApp.$message({
            type: 'error',
            message: `Unknown Backend Error`
          });
    });
}

function sendQuestion(question){
    fetch(`${window.location.href}save_message`,{
        method: 'POST',
        headers: headers,
        body: JSON.stringify({
            'question':question
        }),
        mode: 'cors'
    })
    .then(response => response.json())
    .then(res => {
        console.log('res',res)
        getAnswer(question)
        window.vueApp.messages = res.msglist
    })
    .catch(error => {
        console.error('Error fetching data:', error);
        window.vueApp.isGeneratingAnswer = false
        window.vueApp.$message({
            type: 'error',
            message: `Unknown Backend Error`
          });
    });
}

function generateCaption(newContext){
    fetch(`${window.location.href}caption_gen`,{
        method: 'POST',
        headers: headers,
        body: JSON.stringify({
            'info':newContext
        }),
        mode: 'cors'
    })
    .then(response => response.json())
    .then(res => {
        console.log('res',res)
        window.vueApp.isGeneratingCaption = false
        window.vueApp.caption = res.caption
    })
    .catch(error => {
        console.error('Error fetching data:', error);
        window.vue.isGeneratingCaption = false
        window.vueApp.$message({
            type: 'error',
            message: `Unknown Backend Error`
          });
    });
}