let headers = new Headers();
headers.append('Content-Type', 'application/json');
headers.append('Accept', 'application/json');


// function get_chart(index) {
//     fetch(`${window.location.href}get_org_instance_chart`, {
//         method: 'POST',
//         body: JSON.stringify({
//            'index':index
//         }),
//         headers: headers,
//         mode: 'cors'
//     })
//     .then(response => response.json())
//     .then(res => {
//         window.vueApp.value_list = res.value_list.map(value => parseFloat(value)); /
//         window.vueApp.features_with_bound = res.features_with_bound
//         window.vueApp.org_instance_chart_url = res.img_path
       
//     })
//     .catch(error => {
//         console.error('Error fetching data:', error);
       
//     });
// }
