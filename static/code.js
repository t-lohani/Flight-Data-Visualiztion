var fontFamily = 'verdana';
var width = 1000;
var height = 450;

function onLoad() {
    console.log("Tarun", "Inside onLoad");
    document.getElementById("reductionType").selectedIndex = 0
}

function selectReductionType() {
    console.log("Tarun", "Inside selectReductionType");
    var dropdown = document.getElementById("reductionType");
    var radiobutton = document.getElementById("samplingType");
    var selectedValue = dropdown.options[dropdown.selectedIndex].value;

    if(selectedValue == -1) {
        radiobutton.style.visibility = "hidden";
        // Remove everything
    } else if(selectedValue == "PCA") {
        radiobutton.style.visibility = "visible";
        if (radiobutton[0].checked) {
            draw_pca("/pca_random", true);
        } else {
            draw_pca("/pca_adaptive", false);
        }
    } else if(selectedValue == "ISOMAP") {
        radiobutton.style.visibility = "visible";
        if (radiobutton[0].checked) {
            draw_isomap("/isomap_random", true);
        } else {
            draw_isomap("/isomap_adaptive", false);
        }
    } else if(selectedValue == "MDS_EUCLIDEAN") {
        radiobutton.style.visibility = "visible";
        if (radiobutton[0].checked) {
            draw_mds("/mds_euclidean_random", true);
        } else {
            draw_mds("/mds_euclidean_adaptive", false);
        }
    } else if(selectedValue == "MDS_COSINE") {
        radiobutton.style.visibility = "visible";
        if (radiobutton[0].checked) {
            draw_mds("/mds_cosine_random", true);
        } else {
            draw_mds("/mds_cosine_adaptive", false);
        }
    } else if(selectedValue == "MDS_CORRELATION") {
        radiobutton.style.visibility = "visible";
        if (radiobutton[0].checked) {
            draw_mds("/mds_correlation_random", true);
        } else {
            draw_mds("/mds_correlation_adaptive", false);
        }
    } else if(selectedValue == "LSA") {
        radiobutton.style.visibility = "hidden";
        draw_lsa("/lsa")
    }
}

function selectSampling() {
    selectReductionType();
}

function draw_pca(url, random_sampling) {
    console.log("Tarun", "draw_pca : " + url);
    $.ajax({
	  type: 'GET',
	  url: url,
      contentType: 'application/json; charset=utf-8',
	  xhrFields: {
		withCredentials: false
	  },
	  headers: {

	  },
	  success: function(result) {
		    draw_scatter_plot(result, random_sampling);
//		    drawScreePlot(result)
	  },
	  error: function(result) {
		$("#chart_container").html(result);
	  }
	});
}

function draw_isomap(url, random_sampling) {
    console.log("Tarun", "draw_isomap : " + url);
    $.ajax({
	  type: 'GET',
	  url: url,
      contentType: 'application/json; charset=utf-8',
	  xhrFields: {
		withCredentials: false
	  },
	  headers: {

	  },
	  success: function(result) {
	      draw_scatter_plot(result, random_sampling);
	  },
	  error: function(result) {
		$("#chart_container").html(result);
	  }
	});
}

function draw_mds(url, random_sampling) {
    console.log("Tarun", "draw_mds : " + url);
    $.ajax({
	  type: 'GET',
	  url: url,
      contentType: 'application/json; charset=utf-8',
	  xhrFields: {
		withCredentials: false
	  },
	  headers: {

	  },
	  success: function(result) {
		    draw_scatter_plot(result, random_sampling);
	  },
	  error: function(result) {
		$("#chart_container").html(result);
	  }
	});
}

function draw_lsa(url) {
    console.log("Tarun", "draw_lsa : " + url);
    $.ajax({
	  type: 'GET',
	  url: url,
      contentType: 'application/json; charset=utf-8',
	  xhrFields: {
		withCredentials: false
	  },
	  headers: {

	  },
	  success: function(result) {
            draw_lsa_plot(result);
	  },
	  error: function(result) {
		$("#chart_container").html(result);
	  }
	});
}

function resetEverything() {
    d3.select('#chart').remove();
}

function draw_scatter_plot(chart_data, random_sampling) {
//    console.log("Tarun", "Inside draw_scatter_plot");
    d3.select('#chart').remove();
    var data = JSON.parse(chart_data);
    var obj_array = [];
    var min = 0, max = 0;

    for(var i=0; i < Object.keys(data[0]).length; ++i){
        obj = {}
        obj.x = data[0][i];
        obj.y = data[1][i];
        obj.clusterid = data['clusterid'][i]
        obj.arrival = data['arrival'][i]
        obj.departure = data['departure'][i]
        obj_array.push(obj);
    }

    data = obj_array;

    var margin = {top: 20, right: 20, bottom: 30, left: 40};
    var width = 960 - margin.left - margin.right;
    var height = 500 - margin.top - margin.bottom;

    var xValue = function(d) { return d.x;};
    var xScale = d3.scale.linear().range([0, width]);
    var xMap = function(d) { return xScale(xValue(d)); };
    var xAxis = d3.svg.axis().scale(xScale).orient("bottom");

    var yValue = function(d) { return d.y;};
    var yScale = d3.scale.linear().range([height, 0]);
    var yMap = function(d) { return yScale(yValue(d));};
    var yAxis = d3.svg.axis().scale(yScale).orient("left");

    var cluster_color

    if(random_sampling) {
        cluster_color = function(d) { return d.clusteridx;}
    } else {
        cluster_color = function(d) { return d.clusterid;}
    }
    var color = d3.scale.category10();

    var svg = d3.select("body").append("svg")
        .attr('id', 'chart')
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    var tooltip = d3.select("body").append('div').style('position','absolute');

    xScale.domain([d3.min(data, xValue)-1, d3.max(data, xValue)+1]);
    yScale.domain([d3.min(data, yValue)-1, d3.max(data, yValue)+1]);

    svg.append("g")
          .attr("transform", "translate(0," + height + ")")
          .attr("class", "x axis")
          .call(xAxis)
          .append("text")
          .attr("class", "label")
          .attr("y", -6)
          .attr("x", width)
          .text("x")
          .style("text-anchor", "end");

    svg.append("g")
          .attr("class", "y axis")
          .call(yAxis)
          .append("text")
          .attr("class", "label")
          .attr("y", 6)
          .attr("transform", "rotate(-90)")
          .attr("dy", ".71em")
          .text("y")
          .style("text-anchor", "end");

    svg.selectAll(".dot")
          .data(data)
          .enter().append("circle")
          .attr("class", "dot")
          .attr("cx", xMap)
          .attr("r", 3.5)
          .attr("cy", yMap)
          .style("fill", function(d) { return color(cluster_color(d));})
          .on("mouseover", function(d) {
              tooltip.transition().style('opacity', .9).style(
							'font-family', fontFamily).style('color','steelblue')
              tooltip.html("Departure: " + d.departure + ", Arrival: " + d.arrival)
                   .style("top", (d3.event.pageY - 28) + "px")
                   .style("left", (d3.event.pageX + 5) + "px");
          })
          .on("mouseout", function(d) {
              tooltip.transition()
                   .duration(500)
                   .style("opacity", 0);
              tooltip.html('');
          });
}


function draw_lsa_plot(chart_data, random_sampling) {
    d3.select('#chart').remove();
    var data = JSON.parse(chart_data);
    var min = 0, max = 0;
    var no_of_clusters = data.length;

    var margin = {top: 20, right: 20, bottom: 30, left: 40};
    var circle_radius = 100;

    var color = ['steelblue', '#E69F9F', '#88C37C', '#EEDDCC'];

    var svg = d3.select("body").append("svg")
        .attr('id', 'chart')
        .attr("height", height + margin.top + margin.bottom)
        .attr("width", width + margin.left + margin.right)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    var tooltip = d3.select("body").append("div")
        .attr("class", "tooltip")
        .style("opacity", 0);

    var item = [];
    var start = circle_radius, end, unit = width/no_of_clusters/2;
    for (var i = 0; i < no_of_clusters; i++) {

        var randX = start;
        var randY = Math.floor((Math.random() * (300 - circle_radius)) + circle_radius);
        start += unit + circle_radius;

        svg.append("circle")
          .attr("class", "dot")
          .attr("r", circle_radius)
          .attr("cx", randX)
          .attr("cy", randY)
          .style("fill", color[i])
          .on("mouseover", function(d) {
              tooltip.transition()
                   .duration(200)
                   .style("opacity", .9);
              tooltip.html(d.x)
                   .style("left", (d3.event.pageX + 5) + "px")
                   .style("top", (d3.event.pageY - 28) + "px");
          })
          .on("mouseout", function(d) {
              tooltip.transition()
                   .duration(500)
                   .style("opacity", 0);
          });

          item = data[i];
          for(var j = 0; j < item.length; j++) {
            var v1X = (randX + circle_radius), v1Y = (randX - circle_radius);
            var v2X = (randY + circle_radius), v2Y = (randY - circle_radius);
            var rX = Math.random() * (v1X - v1Y) + v1Y;
            var rY = Math.random() * (v2X - v2Y) + v2Y;
            svg.append("text")
                .attr("x", rX)
                .attr("y", rY)
                .attr("dy", ".35em")
                .text(item[j])
                .style('font', '11px arial');
          }

    }

    var xValue = function(d) { return d.x;}, xScale = d3.scale.linear().range([0, width]),
        xMap = function(d) { return xScale(xValue(d));}, xAxis = d3.svg.axis().scale(xScale).orient("bottom");

    var yValue = function(d) {return d.y;},
        yScale = d3.scale.linear().range([height, 0]),
        yMap = function(d) { return yScale(yValue(d));}, yAxis = d3.svg.axis().scale(yScale).orient("left");

    svg.append("g")
      .attr("transform", "translate(0," + height + ")")
      .call(xAxis)
      .append("text")
      .attr("class", "label")
      .attr("x", width)
      .attr("y", -6)
      .text("x")
      .style("text-anchor", "end");

    svg.append("g")
          .call(yAxis)
          .append("text")
          .attr("class", "label")
          .attr("transform", "rotate(-90)")
          .attr("y", 6)
          .attr("dy", ".71em")
          .text("y")
          .style("text-anchor", "end");
}