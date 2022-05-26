$('#variabel').on('change',function(){

    $.ajax({
        url: "/plotting",
        type: "GET",
        contentType: 'application/json;charset=UTF-8',
        data: {
            'selected': document.getElementById('variabel').value

        },
        dataType:"json",
        success: function (data) {
            Plotly.newPlot('chart1', data );
        }
    });
})