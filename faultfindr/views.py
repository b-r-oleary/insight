import sys
sys.path.insert(0, './faultfindr')

from product_results import ProductResults, Laptops

from flask import render_template, request
from faultfindr import app


laptop_list = Laptops()


@app.route('/')
@app.route('/index')
def laptop_input():
	search_term = request.args.get('srch-term')
	asin        = request.args.get('asin')
	if search_term == 'None':
		search_term = None

	laptops = laptop_list.search_results(search_term)

	if asin is not None:
		results  = ProductResults(asin)

		laptop   = laptop_list.get_details(asin)
		examples = results.formatted_examples
		related  = results.get_related()
		barchart = results.get_review_number_barchart_data()
		rate     = results.review_rates
		piechart = results.get_pie_chart_discussion_data()

	else:
		laptop   = None
		examples = None
		related  = None
		barchart = None
		rate     = None
		piechart = None

	return render_template("index.html",
						   laptops=laptops,
						   search_term=search_term,
						   laptop=laptop,
						   examples=examples,
						   related=related,
						   asin=asin,
						   barchart=barchart,
						   rate=rate,
						   piechart=piechart)