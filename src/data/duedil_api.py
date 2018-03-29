import requests

class DuedilAPI():
	""" Basic usage of DueDil's API."""
	def __init__(self, API_KEY):
		self.API_KEY = API_KEY
		self.headers = {'X-AUTH-TOKEN':API_KEY}
	
	def get_info(self, company_id):
		"""Get company info (as shown in Companies House).

		Args:
			company_id (int): Company ID from Companies House.

		Return:
			JSON with company information.

		"""
		r = requests.get('https://duedil.io/v4/company/gb/{}.json'.format(company_id), headers=self.headers)
		return r.json()
	
	def get_industries(self, company_id):
		r = requests.get('https://duedil.io/v4/company/gb/{}/industries.json'.format(company_id), headers=self.headers)
		return r.json()
	
	def get_description(self, company_id):
		r = requests.get('https://duedil.io/v4/company/gb/{}/descriptions.json'.format(company_id), headers=self.headers)
		return r.json()

	def get_keywords(self, company_id):
		r = requests.get('https://duedil.io/v4/company/gb/{}/keywords.json'.format(company_id), headers=self.headers)
		return r.json()

	def get_social_media(self, company_id):
		r = requests.get('https://duedil.io/v4/company/gb/{}/social-media-profiles.json'.format(company_id), headers=self.headers)
		return r.json()
