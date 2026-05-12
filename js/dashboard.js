let model;
let chart;

async function loadModel() {

    model = await tf.loadLayersModel("./model/model.json");

    document.getElementById("status").innerText =
        "Model loaded successfully!";
}

async function runPrediction() {

    if (!model) {
        alert("Model still loading!");
        return;
    }

    // 예시 입력 데이터
    // 나중에 실제 데이터로 바꾸면 됨
    const inputSeries = [];

    for (let i = 0; i < 72; i++) {
        inputSeries.push(Math.sin(i / 10));
    }

    // INPUT SHAPE 중요
    // 네 모델 shape에 맞춰야 함
    const inputTensor =
        tf.tensor(inputSeries).reshape([1, 72]);

    // 예측
    const prediction = model.predict(inputTensor);

    const output = await prediction.array();

    console.log(output);

    drawChart(inputSeries, output[0]);

    inputTensor.dispose();
    prediction.dispose();
}

function drawChart(history, forecast) {

    const ctx =
        document.getElementById('forecastChart');

    if (chart) {
        chart.destroy();
    }

    const labels = [];

    for (let i = 0; i < history.length + forecast.length; i++) {
        labels.push(i);
    }

    const historyData =
        history.concat(Array(forecast.length).fill(null));

    const forecastData =
        Array(history.length).fill(null).concat(forecast);

    chart = new Chart(ctx, {

        type: 'line',

        data: {
            labels: labels,

            datasets: [
                {
                    label: 'Input Series',
                    data: historyData
                },
                {
                    label: 'Forecast',
                    data: forecastData
                }
            ]
        }
    });
}

loadModel();
