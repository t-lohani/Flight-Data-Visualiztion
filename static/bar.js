function selectReductionType() {
    var dropdown = document.getElementById("reductionType");
    var radiobutton = document.getElementById("samplingType");
    var selectedValue = dropdown.options[dropdown.selectedIndex].value;
    console.log("Tarun", radiobutton[1].checked);
    if(selectedValue == -1) {
        // Do nothing
    } else if(selectedValue == "PCA") {
//        get_map('/pca_random', true, false, true);
    } else if(selectedValue == "ISOMAP") {
//        get_map('/isomap_random', true, false, false);
    } else if(selectedValue == "MDS_EUCLIDEAN") {
//        get_map('/mds_euclidean_random', true, false, false);
    } else if(selectedValue == "MDS_COSINE") {
//        get_map('/mds_cosine_random', true, false, false);
    } else if(selectedValue == "MDS_CORRELATION") {
//        get_map('/mds_correlation_random', true, false, false);
    } else if(selectedValue == "LATENT_SEMANTIC_ANALYSIS") {
//        get_map('/lsa', false, true, false);
    }
//    d3.select('#scree').remove();
}


function get_map(url, rs, lsa, isPCA) {
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
	    if(lsa) {
		    drawLSA(result, rs);
		} else {
		    drawScatter(result, rs);
		    if(isPCA) {
		        drawScreePlot(result)
		    }
		}
	  },
	  error: function(result) {
		$("#body1").html(result);
	  }
	});
}