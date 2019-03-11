## happy path
* greet
  - utter_greet

## say goodbye
* bye
  - utter_bye

## book room 0
* book_room{"room_type": "r"}
  - book_room
  - form{"name": "book_room"}
  - form{"name": null}

## book room 1
* book_room{"room_type": "r"}
  - book_room
  - slot{"room_type": "r"}
  - slot{"requested_slot": "guest_name"}
  - utter_ask_guest_name
* info{"guest_name": "r"}
  - book_room
  - slot{"guest_name": "r"}
  - slot{"requested_slot": "guest_phone_number"}
  - utter_ask_guest_phone_number
* info{"guest_phone_number": "18088880000"}
  - book_room
  - slot{"guest_phone_number": "18088880000"}
  - slot{"requested_slot": "checkin_time"}
  - utter_ask_checkin_time
* info{"checkin_time": "today"}
  - book_room
  - slot{"checkin_time": "today"}
  - slot{"requested_slot": "confirmed"}
  - utter_ok
  - ask_confirm
* confirm
  - book_room
  - slot{"confirmed": true}
  - utter_bye

