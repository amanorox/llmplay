import {
    AutoModel,
    AutoTokenizer,
    env,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.1.2';

env.backends.onnx.wasm.numThreads = 8;

const MODEL_NAME = "onnx-community/Qwen2.5-0.5B";
let g_chart;
let g_model;
let g_tokenizer;

function debugprint( text ){
    document.querySelector( "#debug" ).innerText = text;
}


async function loadModel(){
    let loading_progress = {};
    g_tokenizer = await AutoTokenizer.from_pretrained( MODEL_NAME );
    let progress_str;
    g_model = await AutoModel.from_pretrained( MODEL_NAME, {
        dtype: "fp16",
        progress_callback: ( progress ) => {
            console.log( progress )
            if( progress.status == "progress" ){
                loading_progress[progress.file] = progress;
                progress_str = "Now loading...\n";
                Object.keys( loading_progress ).forEach( ( v ) => {
                    progress_str += `${loading_progress[v].file} ${loading_progress[v].progress}%\n`;
                } );
                debugprint( progress_str );
            }
        }
    } );
    progress_str += "Done.";
    debugprint( progress_str );
}

const labels = [
    "明日の天気は",
    "私の住んでいる",
    "日本の将来を",
    "税金が高いので",
    "昔々あるところに"
];
const data = {
    labels: labels,
    datasets: [
        {
            label: '次の語句を選択',
            data: [1.0, 1.0, 1.0, 1.0, 1.0],
            borderColor: "red",
            backgroundColor: "#ff000080",
        },
    ]
};

function softmax( arr ){
    const exponents = arr.map( Math.exp ),
        total = exponents.reduce( ( a, b ) => a + b, 0 );
    return exponents.map( ( exp ) => exp / total );
}

async function predictNextToken( text ){
    let inputs = await g_tokenizer( text, {add_special_tokens: false} );
    let {logits} = await g_model( inputs );

    logits = logits.slice( null, -1, null );
    let probs = softmax( logits.data );
    let indexedArray = Array.from( probs, ( value, index ) => ({value, index}) );
    indexedArray.sort( ( a, b ) => b.value - a.value );
    let sortedIndexes = indexedArray.map( item => item.index );

    let labels = [];
    let values = [];
    for( let i = 0; i < 50; i++ ){
        let token = sortedIndexes[i];
        let s = g_tokenizer.decode( [token], {skip_special_tokens: true} );
        labels.push( s );
        values.push( probs[token] );
    }
    g_chart.data.labels = labels;
    g_chart.data.datasets[0].data = values;
    g_chart.update();
}

window.addEventListener( "load", async () => {
    await loadModel();

    document.querySelector( "#btn-start" ).addEventListener( "click", async () => {
        let text = document.querySelector( "#start-text" ).value;
        document.querySelector( "#output" ).textContent = text;
        predictNextToken( text );
    } );

    const config = {
        type: 'bar',
        data: data,
        options: {
            scales: {
                x: {
                    position: 'top',
                },
            },
            indexAxis: 'y',
            responsive: false,
            onClick: ( event ) => {
                const points = g_chart.getElementsAtEventForMode( event, 'y', {intersect: false}, true );
                console.log( `XY=${event.x}, ${event.y}` );
                if( points.length ){
                    const firstPoint = points[0];
                    const label = g_chart.data.labels[firstPoint.index];
                    console.log( `Label: ${label}` );
                    let text = document.querySelector( "#output" ).textContent;
                    text += label;
                    document.querySelector( "#output" ).textContent = text;
                    predictNextToken( text );
                }
            }
        },
    };
    const ctx = document.querySelector( "#llm-next-words" );
    g_chart = new Chart( ctx, config );
} );