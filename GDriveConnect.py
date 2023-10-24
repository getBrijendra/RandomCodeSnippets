import os
import traceback

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow, InstalledAppFlow
from googleapiclient.discovery import build
import json

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
API_SERVICE_NAME = 'drive'
API_VERSION = 'v3'


# @knowledge_app.route(Config.G_DRIVE_AUTH_URI, methods=['GET'])
# @token_required_v2
def get_gdrive_auth_uri(body):
    """
    Generate the Google Drive OAuth2.0 authorization URL.

    Returns:
        tuple: A tuple containing a JSON response and an HTTP status code.

    This route is used to generate the Google Drive OAuth2.0 authorization URL for a specific tenant. It requires
    a tenant key and a 'redirect_uri' as a query parameter.

    Args:
        - 'redirect_uri' (str): The redirect URI for handling the OAuth2 callback. If not provided, a
          'bad_request' response is returned.

    Upon successful execution, the function generates the authorization URL and responds with a JSON object
    containing the URL. If an error occurs, it returns an error response with an appropriate status code.
    """
    redirect_uri = body.get('redirect_uri', None)
    if not redirect_uri:
        return bad_request(message="Missing redirect URI")

    try:
        # Create flow instance to manage the OAuth 2.0 Authorization Grant Flow steps.
        flow = Flow.from_client_secrets_file(
            Config.DRIVE_CLIENT_SECRETS_FILE, scopes=SCOPES)

        # This will be the URI where we will get the authorization code
        flow.redirect_uri = redirect_uri

        authorization_url, state = flow.authorization_url(
            access_type='offline',
            # Enable incremental authorization. Recommended as a best practice.
            include_granted_scopes='true')

        return success(data={'authorization_url': authorization_url})
    except Exception as e:
        traceback.print_exc()
        print(e)
        return internal_server_error(message=str(e))


# @knowledge_app.route(Config.G_DRIVE_OAUTH_2_CALLBACK, methods=['GET'])
# @token_required_v2
def oauth2callback(body):
    """
    Handle the OAuth2 callback and save Google Drive credentials in the database.

    Args:
        tenant_key (str): The key identifying the tenant in the database.

    Returns:
        tuple: A tuple containing a message and an HTTP status code.

    This route is used to handle the OAuth2 callback from Google Drive. It expects the following query parameters:
    
    - 'scope' (str): The authorized scope for the access token.
    - 'code' (str): The authorization code received from Google Drive.
    - 'redirect_uri' (str): The redirect URI used for the OAuth2 flow.

    Upon successful authentication and authorization, the function fetches the OAuth2 tokens, saves them in the database,
    and responds with a message indicating successful validation.
    """
    tenant_key = body.get('tenantId', None)
    if not tenant_key:
        return bad_request(message="Missing tenant_key")
    AUTHORIZED_SCOPES = body.get('scope')
    if not AUTHORIZED_SCOPES:
        return bad_request(message="Missing scope")
    auth_code = body.get('code')
    if not auth_code:
        return bad_request(message="Missing code")
    redirect_uri = body.get('redirect_uri')
    if not redirect_uri:
        return bad_request(message="Missing redirect_uri")
    print(AUTHORIZED_SCOPES)

    try:
        flow = Flow.from_client_secrets_file(
            Config.DRIVE_CLIENT_SECRETS_FILE, scopes=AUTHORIZED_SCOPES)
        flow.redirect_uri = redirect_uri

        # Use the authorization server's response to fetch the OAuth 2.0 tokens.
        flow.fetch_token(code=auth_code)

        credentials = credentials_to_dict((flow.credentials))
        print(credentials)

        # Saving credentials in DB
        update_operation = {
            '$set': {
                'gDriveCreds': credentials
            }
        }
        filter_condition = {
            'key': int(tenant_key)
        }
        result = db.collection.update_one(
            filter_condition, update_operation)
        print(f"Modified {result.modified_count} document")

        return success(message="Credentials created successfully.")
    except Exception as e:
        traceback.print_exc()
        print(e)
        return internal_server_error(message=str(e))


# @knowledge_app.route(Config.G_DRIVE_REVOKE_ACCESS, methods=['GET'])
# # @token_required_v2
# def revoke_gdrive_access(tenant_key):
#     pass


# @knowledge_app.route(Config.G_DRIVE_LIST_FOLDERS, methods=['GET'])
# @token_required_v2
def list_gdrive_folders(body):
    """
    List folders in a user's Google Drive.

    Args:
        tenant_key (str): The key identifying the tenant in the database.

    Returns:
        tuple: A tuple containing a JSON response and an HTTP status code.

    This route allows you to list folders in a user's Google Drive. It requires a tenant key
    to fetch the user's Google Drive credentials from the database. You can optionally specify
    a 'folderId', 'pageSize', and 'nextPageToken' as query parameters to customize the folder listing.

    - 'folderId' (str): The ID of the parent folder. Defaults to 'root'.
    - 'pageSize' (int): The number of folders to fetch per request. Defaults to 20.
    - 'nextPageToken' (str): A token for pagination if there are more results.

    The function handles Google Drive API authentication using credentials from the database,
    constructs a query to list only folders, and executes the query to retrieve folder information.
    If successful, it returns a JSON response with folder details. If an error occurs, it provides
    an error response with appropriate status codes.
    """
    folder_id = body.get('folderId', 'root')
    page_size = int(body.get('pageSize', 20))
    next_page_token = body.get('nextPageToken', None)
    tenant_key = body.get('tenantId', None)
    if not tenant_key:
        return bad_request(message="Missing tenant_key")
    try:
        # Get credentials from DB
        tenant = db.collection.find_one({'key': tenant_key})
        credentials = tenant.get('gDriveCreds', None)
        if not credentials:
            return unauthorized(message="Credentials not found. Please give access to Google Drive.")
        creds = Credentials.from_authorized_user_info(credentials)
        # query filter to list only folders
        query = "mimeType='application/vnd.google-apps.folder'"
        query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder'"
        service = build(API_SERVICE_NAME, API_VERSION, credentials=creds)
        results = service.files().list(q=query, pageSize=page_size,
                                    pageToken=next_page_token).execute()
        return success(data=results)
    except Exception as e:
        traceback.print_exc()
        print(e)
        return internal_server_error(message=str(e))
