from twilio.rest import Client

def send_whatsapp_text_message(account_sid, auth_token, from_number, to_number, message):
    client = Client(account_sid, auth_token)

    try:
        message = client.messages.create(
            from_=f'whatsapp:{from_number}',
            body=message,
            to=f'whatsapp:{to_number}',
        )
        print(f"Message sent successfully. SID: {message.sid}")
    except Exception as e:
        print(f"Error sending message: {str(e)}")

    # Example usage
account_sid = 'ACa5e43dc25802687dfe14ac2df99bf5f4'
auth_token = '0b22b5f49f0c0e33412b0ea48edd7ed7'
from_number = '+14155238886'
to_number = '+971529123587'
text_message = 'Today, we have achieved an 80% accuracy rate in predicting the footfall, which amounts to 3,230 visitors. We anticipate the crowd to be more active between 11 AM and 12 PM and between 5 PM and 9 PM. To ensure effective staff management, please be prepared to allocate appropriate resources during these time frames.'
text_message1 = 'There seems to be a significant gathering, with all the points of sale (POS) currently operational. It is advised to initiate the utilization of Q-Buster in the SSM.'

# send_whatsapp_text_message(account_sid, auth_token, from_number, to_number, text_message)
# send_whatsapp_text_message(account_sid, auth_token, from_number, to_number, text_message1)



def send_whatsapp_message(account_sid, auth_token, from_number, to_number, message, media_url):
    client = Client(account_sid, auth_token)

    try:
        message = client.messages.create(
            from_=f'whatsapp:{from_number}',
            body=message,
            media_url=media_url,
            to=f'whatsapp:{to_number}',
        )
        print(f"Message sent successfully. SID: {message.sid}")
    except Exception as e:
        print(f"Error sending message: {str(e)}")