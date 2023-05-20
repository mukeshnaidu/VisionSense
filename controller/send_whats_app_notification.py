from twilio.rest import Client


def sendwhatsappnotification(image):
    try:
        print("Separate thread is running")
        account_sid = 'AC9a3acd4165bb8239204ffa6e619a9951'
        auth_token = 'af23eb36c51eccbf0cb531949c40bcc9'
        client = Client(account_sid, auth_token)

        message = client.messages.create(
            from_='whatsapp:+14155238886',
            media_url="https://ichef.bbci.co.uk/images/ic/1920x1080/p04vbgvg.jpg",
            body='how are you',
            to='whatsapp:+971529123587'
        )
        print(message.sid)
    except:
        print("Something went wrong")
    finally:
        print("The 'try except' is finished")
