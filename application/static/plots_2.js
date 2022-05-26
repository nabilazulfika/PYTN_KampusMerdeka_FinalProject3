$('#model').on('change',function(){

    $.ajax({
        url: "/chart",
        type: "GET",
        contentType: 'application/json;charset=UTF-8',
        data: {
            'terpilih': document.getElementById('model').value

        },
        dataType:"json",
        success: function (roc) {
            Plotly.newPlot('chart1', roc);
        }
    });
})
