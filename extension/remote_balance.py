from modules.shared import log

from extension.utils_remote import get_current_api_service, RemoteService, request_or_error, get_api_key, stable_horde_client

balance_names = {
    RemoteService.StableHorde: ('Kudos', ''),
    RemoteService.OmniInfer: ('Credits', '$')
}

def get_remote_balance_html():
    service = get_current_api_service()

    balance = get_remote_balance(service)
    log.debug(f'RI: {service} balance: {balance}')

    if balance is None:
        return ''
    title, symbol = balance_names[service]

    return f'<p>{title}:</p><p id="remote_inference_balance_count">{symbol}{balance}</p>'

def get_remote_balance(service):
    if service == RemoteService.StableHorde:
        response = request_or_error(service, '/v2/find_user', headers={"apikey": get_api_key(service), "Client-Agent": stable_horde_client})
        return int(response['kudos'])
    elif service == RemoteService.OmniInfer:
        response = request_or_error(service, '/v3/user', headers={"X-Omni-Key": get_api_key(service)})
        return response['credit_balance']/10000.

    return None