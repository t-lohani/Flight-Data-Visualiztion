var fontFamily = 'verdana';
var width = 1366;
var height = 400;
var stroke_color = '#FFF';

function onLoadTask2b() {
//    console.log("Tarun", "Inside onLoadTask2b");
    document.getElementById("samplingTypeScree")[0].checked = true;
    selectScreeType()
}

function onLoadTask3() {
//    console.log("Tarun", "Inside onLoadTask3");
    document.getElementById("samplingType")[0].checked = true;
    document.getElementById("reductionType").selectedIndex = 0;
}

function onLoadTask2c() {
    console.log("Tarun", "Inside onLoadTask2c");
    url = "/get_squareloadings"
    console.log("Tarun", "getSquareLoadings : " + url);
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
	        draw_histogram(result)
	  },
	  error: function(result) {
		$("#chart_container").html(result);
	  }
	});
}

function selectScreeType() {
    if (document.getElementById("samplingTypeScree")[0].checked) {
        draw_scree("/scree_plot_random", "Scree Plot - PCA Random samples");
    } else {
        draw_scree("/scree_plot_adaptive", "Scree Plot - PCA Adaptive samples");
    }
}

function selectReductionType() {
//    console.log("Tarun", "Inside selectReductionType");
    var dropdown = document.getElementById("reductionType");
    var radiobutton = document.getElementById("samplingType");
    var selectedValue = dropdown.options[dropdown.selectedIndex].value;

    if(selectedValue == -1) {
        radiobutton.style.visibility = "hidden";
        resetEverything()
        // Remove everything
    } else if(selectedValue == "PCA") {
        radiobutton.style.visibility = "visible";
        if (radiobutton[0].checked) {
            draw_scatter("/pca_random", true, "Scatter plot for PCA (Random Sampling)");
        } else {
            draw_scatter("/pca_adaptive", false, "Scatter plot for PCA (Adaptive Sampling)");
        }
    } else if(selectedValue == "MDS_EUCLIDEAN") {
        radiobutton.style.visibility = "visible";
        if (radiobutton[0].checked) {
            draw_scatter("/mds_euclidean_random", true, "Scatter plot for MDS using Euclidean distances (Random Sampling)");
        } else {
            draw_scatter("/mds_euclidean_adaptive", false, "Scatter plot for MDS using Euclidean distances (Adaptive Sampling)");
        }
    } else if(selectedValue == "MDS_CORRELATION") {
        radiobutton.style.visibility = "visible";
        if (radiobutton[0].checked) {
            draw_scatter("/mds_correlation_random", true, "Scatter plot for MDS using Correlation (Random Sampling)");
        } else {
            draw_scatter("/mds_correlation_adaptive", false, "Scatter plot for MDS using Correlation (Adaptive Sampling)");
        }
    } else if(selectedValue == "SCATTERPLOT") {
        radiobutton.style.visibility = "visible";
        if (radiobutton[0].checked) {
            draw_scatter_matrix("/scatterplot_matrix_random", true, "Scatterplot Matrix for top 3 PCA loaded attributes (Random Sampling)");
        } else {
            draw_scatter_matrix("/scatterplot_matrix_adaptive", false, "Scatterplot Matrix for top 3 PCA loaded attributes (Adaptive Sampling)");
        }
    }
}

function draw_scatter(url, random_sampling, plot_text) {
//    console.log("", "draw_scatter : " + url);
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
		    draw_scatter_plot(result, random_sampling, plot_text);
	  },
	  error: function(result) {
		$("#chart_container").html(result);
	  }
	});
}

function draw_scree(url, plot_text) {
//    console.log("Tarun", "draw_scree : " + url);
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
		    draw_scree_plot(result, plot_text);
	  },
	  error: function(result) {
		$("#chart_container").html(result);
	  }
	});
}

function draw_scatter_matrix(url, random_sampling, plot_text) {
//    console.log("Tarun", "draw_scatter_matrix : " + url);
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
		    draw_scatter_matrix_plot(result, random_sampling, plot_text);
	  },
	  error: function(result) {
		$("#chart_container").html(result);
	  }
	});
}

function resetEverything() {
    d3.select('#chart').remove();
}

function draw_histogram(data) {
    console.log("Tarun", "Inside draw_histogram")
    d3.select('#barchart').remove();
    loadingVector = JSON.parse(data);

    var feature_names = Object.keys(loadingVector);
//    console.log("Tarun", feature_names);

    var feature_loadings = []

    for (var i=0; i<feature_names.length; i++) {
        feature_loadings[i] = loadingVector[feature_names[i]];
    }

//    console.log("Tarun", feature_loadings);
    var width = 1000;
    var bar_height = 50;
    var padding = 6;
    var left_width = 160;
    var height = (bar_height + padding) * feature_names.length;
    var chart_height = 540;

    chart = d3.select("body")
      .append('svg')
      .attr('id', 'histogram')
      .attr('class', 'barchart')
      .attr('width', left_width + width + 40)
      .attr('height', chart_height)
      .append("g")
      .style("transform", "translate(0px, 30px)");

    var x = d3.scale.linear()
       .domain([0, d3.max(feature_loadings)])
       .range([0, width - 200]);

    var y = d3.scale.ordinal()
        .domain(feature_loadings)
        .rangeBands([0, (bar_height + 2 * padding) * feature_loadings.length]);

    var y2 = d3.scale.ordinal()
        .domain(feature_names)
        .rangeBands([0, (bar_height + 2 * padding) * feature_names.length]);

    var line = chart.selectAll("line")
       .data(x.ticks(10))
       .enter().append("line")
       .attr("class", "barline")
       .attr("x1", function(d) { return x(d) + left_width; })
       .attr("x2", function(d) { return x(d) + left_width; })
       .attr("y1", 0)
       .attr("y2", (bar_height + padding * 2) * feature_names.length);

    var rule = chart.selectAll(".rule")
       .data(x.ticks(10))
       .enter().append("text")
       .attr("class", "barrule")
       .attr("x", function(d) { return x(d) + left_width; })
       .attr("y", 0)
       .attr("dy", -6)
       .attr("text-anchor", "middle")
       .attr("font-size", 10)
       .text(String);

    var rect = chart.selectAll("rect")
       .data(feature_loadings)
       .enter().append("rect")
       .attr("x", left_width)
       .attr("y", function(d) { return y(d) + padding; })
       .attr("width", x)
       .attr("height", bar_height)
//       .duration(1000).delay(300).ease('elastic');

    var loadings = chart.selectAll("loadings")
       .data(feature_loadings)
       .enter().append("text")
       .attr("x", function(d) { return x(d) + left_width; })
       .attr("y", function(d){ return y(d) + y.rangeBand()/2; })
       .attr("dx", 135)
       .attr("dy", ".36em")
       .attr("text-anchor", "end")
       .attr('class', 'loadings')
       .text(String);

    var names = chart.selectAll("names")
       .data(feature_names)
       .enter().append("text")
       .attr("x", 0)
       .attr("y", function(d){ return y2(d) + y.rangeBand()/2; } )
       .attr("dy", ".36em")
       .attr("text-anchor", "start")
       .attr('class', 'names')
       .text(String);

    chart.append("text")
        .attr("class", "hish_name")
        .attr("text-anchor", "middle")
        .text("PCA loadings of top seven attributes");
}

function draw_scatter_plot(chart_data, random_sampling, plot_text) {
//    console.log("Tarun", "Inside draw_scatter_plot");
    d3.select('#chart').remove();
    var data = JSON.parse(chart_data);
    var obj_array = [];
    var min = 0, max = 0;

    feature_names = Object.keys(data);

    for(var i=0; i < Object.keys(data[0]).length; ++i){
        obj = {}
        obj.x = data[0][i];
        obj.y = data[1][i];

        obj.clusterid = data['clusterid'][i]
        obj.col1 = data[feature_names[2]][i]
        obj.col2 = data[feature_names[3]][i]
        obj_array.push(obj);
    }

    data = obj_array;

    var margin = {top: 20, right: 20, bottom: 30, left: 60};
    var width = 1300 - margin.left - margin.right;
    var height = 450 - margin.top - margin.bottom;

    var chart_width = 800;
    var chart_height = 550;

    var xValue = function(d) { return d.x;};
    var xScale = d3.scale.linear().range([0, chart_width]);
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
        .attr("height", chart_height)
        .append("g")
        .attr("transform", "translate(250,10)");

    var tooltip = d3.select("body").append('div').style('position','absolute');

    xScale.domain([d3.min(data, xValue)-1, d3.max(data, xValue)+1]);
    yScale.domain([d3.min(data, yValue)-1, d3.max(data, yValue)+1]);

    xAxisLine = svg.append("g")
          .attr("transform", "translate(0," + height + ")")
          .attr("class", "x_axis")
          .call(xAxis)

    yAxisLine = svg.append("g")
          .attr("class", "y_axis")
          .call(yAxis)

    svg.append("text")
            .attr("class", "axis_label")
            .attr("text-anchor", "middle")
            .attr("transform", "translate("+ (-70) +","+(height/2)+")rotate(-90)")
            .text("Component 2");

    svg.append("text")
        .attr("class", "axis_label")
        .attr("text-anchor", "middle")
        .attr("transform", "translate("+ (chart_width/2) +","+(height + margin.top + margin.bottom)+")")
        .text("Component 1");

    svg.append("text")
        .attr("class", "chart_name")
        .attr("text-anchor", "middle")
        .attr("transform", "translate("+ (chart_width/2) +","+(chart_height - margin.top - margin.bottom)+")")
        .text(plot_text);

    svg.selectAll(".dot")
          .data(data)
          .enter().append("circle")
          .attr("class", "dot")
          .attr("cx", xMap)
          .attr("r", 3.5)
          .attr("cy", yMap)
          .style("fill", function(d) { return color(cluster_color(d));})
          .on("mouseover", function(d) {
              tooltip.transition().style('opacity', .9)
                .style('font-family', fontFamily)
                .style('color','white')
                .style('font-size', '10px')
              tooltip.html(feature_names[2] + ":" + d.col1 + ", " + feature_names[3] + ":" + d.col2)
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

function draw_scree_plot(eigen_values, plot_text) {
    console.log("Tarun", "Inside draw_scree_plot");

    var data = JSON.parse(eigen_values);
    d3.select('#chart').remove();

    var margin = {top: 20, right: 20, bottom: 30, left: 60};
    var width = 1300 - margin.left - margin.right;
    var height = 450 - margin.top - margin.bottom;

    var chart_width = 800;
    var chart_height = 550;

    var x = d3.scale.linear().domain([1, data.length + 0.5]).range([0, chart_width - 120]);
    var y = d3.scale.linear().domain([0, d3.max(data)]).range([height, 0]);

    var xAxis = d3.svg.axis().scale(x).orient("bottom").ticks(7);
    var yAxis = d3.svg.axis().scale(y).orient("left");

    var markerX
    var markerY
    var color = d3.scale.category10();

    var line = d3.svg.line()
        .x(function(d,i) {
            if (i == 2) {
                markerX = x(i);
                markerY = y(d)
            }
            return x(i);
        })
        .y(function(d) { return y(d); })

    // Add an SVG element with the desired dimensions and margin.
    var svg = d3.select("body").append("svg")
          .attr("id", "chart")
          .attr("width", width + margin.left + margin.right)
          .attr("height", chart_height)
          .append("g")
          .attr("transform", "translate(250,10)");

    // create yAxis
    svg.append("g")
          .attr("class", "x_axis")
          .attr("transform", "translate(110," + height + ")")
          .call(xAxis);

    // Add the y-axis to the left
    svg.append("g")
          .attr("class", "y_axis")
          .attr("transform", "translate(100,0)")
          .call(yAxis);

    svg.append("text")
        .attr("class", "axis_label")
        .attr("text-anchor", "middle")
        .attr("transform", "translate("+ 50 +","+(height/2)+")rotate(-90)")
        .text("Eigen Values");

    svg.append("text")
        .attr("class", "axis_label")
        .attr("text-anchor", "middle")
        .attr("transform", "translate("+ ((chart_width/2) + 50) +","+ ((chart_width/2) + 50) +")")
        .text("K");

    svg.append("text")
        .attr("class", "chart_name")
        .attr("text-anchor", "middle")
        .attr("transform", "translate("+ (height + margin.top + margin.bottom) +","+(chart_height - margin.bottom)+")")
        .text(plot_text);

    svg.append("path")
        .attr("class", "screepath")
        .attr("d", line(data))
        .attr("transform", "translate(215,0)");

    svg.append("circle")
              .attr("cx", markerX)
              .attr("cy", markerY)
              .attr("r", 6)
              .attr("transform", "translate(215,0)")
              .style("fill", "red")
              .style("stroke", "red")

    console.log("Tarun", markerY)
    var th_line = svg.append("line")
        .attr("x1", 100)
        .attr("y1", markerY)
        .attr("x2", 785)
        .attr("y2", markerY)
        .attr("stroke-width", 2)
        .attr("stroke", "green")
}

function draw_scatter_matrix_plot(chart_data, random_sampling, plot_text) {
    console.log("Tarun", "Inside draw_scatter_matrix_plot");
    d3.select('#chart').remove();
    var jdata = JSON.parse(chart_data);
    var traits = Object.keys(jdata);

    var width = 960;
    var size = 170;
    var padding = 10;

    var chart_width = 1300;
    var chart_height = 560;

    var x = d3.scale.linear().range([padding/2, size - padding/2]);
    var y = d3.scale.linear().range([size - padding/2, padding/2]);

    var xAxis = d3.svg.axis().orient("bottom").scale(x).ticks(6);
    var yAxis = d3.svg.axis().orient("left").scale(y).ticks(6);

    var cluster_color

    if(random_sampling) {
            cluster_color = function(d) { return d.clusteridx;}
        } else {
            cluster_color = function(d) { return d.clusterid;}
        }
        var color = d3.scale.category10();
    var color = d3.scale.category10();

    data = {};
    data[traits[0]] = jdata[traits[0]];
    data[traits[1]] = jdata[traits[1]];
    data[traits[2]] = jdata[traits[2]];

    var domainByTrait = {};
    var traits = d3.keys(data).filter(function(d) { return d !== "clusterid"; });
    var n = traits.length;

    xAxis.tickSize(size * n);
    yAxis.tickSize(-size * n);

    traits.forEach(function(trait) {
        domainByTrait[trait] = d3.extent(d3.values(data[trait]));
    });

    var svg = d3.select("body").append("svg")
      .attr("id", "chart")
      .attr("width", chart_width)
      .attr("height", chart_height)
      .append("g")
      .attr("transform", "translate(400,0)");

    svg.selectAll(".x.axis")
      .data(traits)
      .enter().append("g")
      .attr("class", "x_axis_scatter")
      .attr("transform", function(d, i) { return "translate(" + (n - i - 1) * size + ",0)"; })
      .each(function(d) { x.domain(domainByTrait[d]); d3.select(this).call(xAxis); });

     svg.selectAll(".y.axis")
      .data(traits)
      .enter().append("g")
      .attr("class", "y_axis_scatter")
      .attr("transform", function(d, i) { return "translate(0," + i * size + ")"; })
      .each(function(d) { y.domain(domainByTrait[d]); d3.select(this).call(yAxis); });


      var cell = svg.selectAll(".cell")
      .data(cross(traits, traits))
      .enter().append("g")
      .attr("class", "cell")
      .attr("transform", function(d) { return "translate(" + (n - d.i - 1) * size + "," + d.j * size + ")"; })
      .each(plot);

      svg.append("text")
        .attr("class", "scree_name")
        .attr("text-anchor", "middle")
        .text(plot_text);

      cell.filter(function(d) { return d.i === d.j; }).append("text")
      .attr("x", padding)
      .attr("y", padding)
      .attr("dy", ".71em")
      .text(function(d) { return d.x; });


      function plot(p) {
          var cell = d3.select(this);
          x.domain(domainByTrait[String(p.x)]);
          y.domain(domainByTrait[String(p.y)]);

          cell.append("rect")
              .attr("class", "frame")
              .attr("x", padding / 2)
              .attr("y", padding / 2)
              .attr("width", size - padding)
              .attr("height", size - padding);

          first_comp = data[String(p.x)];
          second_comp = data[String(p.y)];
          result_array = []
          second = d3.values(second_comp)
          cluster = jdata['clusterid']

          var count = 0;
          d3.values(first_comp).forEach(function(item, index) {
              temp_map = {};
              temp_map["x"] = item;
              temp_map["y"] = second[index];
              temp_map['cluster'] = cluster[index]
              result_array.push(temp_map);
          });

          cell.selectAll("circle")
              .data(result_array)
              .enter().append("circle")
              .attr("cx", function(d) { return x(d.x); })
              .attr("cy", function(d) { return y(d.y); })
              .attr("r", 4)
              .style("fill", function(d) { return random_sampling ? color(0) : color(d.cluster);});
      }
}

function cross(trait_a, trait_b) {
  var ret_mat = [], len_a = trait_a.length, len_b = trait_b.length, i, j;
  for (i = 0; i < len_a; i++)
    for (j = 0; j < len_b; j++)
      ret_mat.push(
        {x: trait_a[i], i: i,
          y: trait_b[j], j: j});
  return ret_mat;
}