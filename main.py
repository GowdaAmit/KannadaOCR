
import utime
from machine import UART, Pin, Timer, WDT

from collections import OrderedDict
import _thread

led = Pin(25, Pin.OUT)

gsm = machine.UART(1, timeout=1)
EOL = '\r\n'
PASS_THRU_STRS = (b'\n', b'+',b':', b',', b'.', b'=', b' ', b'-')
ACCEPTED_COMMANDS = ['+CLIP', '+CMT', 'ERROR']
DEBUG = False

class Driver :
    
    def __init__(self, gsm, led):
        self._gsm = gsm
        self._led = led
        self._requestQue = [] # stores upto ten requests at any timem rest discarded
        self._que_locked = False
        self._user_list = {} # dictionary of index Phone
        self._in_progress = False
#         self._running_user = None
        self._motor = None
        self._nu_index = 0
        self._admin = None
        self._wdog = None
        self._processReuest = None
        self._motor_timer = Timer()
        self._motor_running = False
        self._motorPower = Pin(18, Pin.OUT)

        
    def __del__(self):
        self._processReuest.deinit()
        del self._motor
        

    class User:

        def __init__(self, pb_index, phone_no, balance):
            
            _balance_rate = balance.split(',')
            
            _balance = _balance_rate[0]
            self._rate = 0.0
            
            if len(_balance_rate)==2:
                self._rate = float(_balance_rate[1])
               
            self._pb_index = pb_index
            self._phone = phone_no
            self._balance = 0 if 'ADMIN' == _balance else float(_balance)
            self._is_admin = True if 'ADMIN' == _balance else False
            

    class Admin(User):

        def __init__(self, index, phone):
            
            Driver.User.__init__(self, index, phone, "ADMIN,0.0")
        

        def _changeAdmin(self, new_admin_no, _driver):
            
            _costPerMinute = _driver._user_list[_driver._admin]._rate
            print("the cost per minute ", _costPerMinute, new_admin_no)
            
            _response = ''
            _new =  False
            #add or update the new admin
            if new_admin_no in _driver._user_list:

                new_adm_index = _driver._user_list[new_admin_no]._pb_index
                _command = "AT+CPBW=" + str(new_adm_index) + ", \"" + new_admin_no  + "\", 129,\"ADMIN," + str(_costPerMinute) + "\""
                _response = _driver.sendCommand(_command)
                
            else:
                # create a new record
                _command = "AT+CPBW=" + str(_driver._nu_index) + ", \"" + new_admin_no + \
                           "\", 129,\"ADMIN," + str(_costPerMinute) + "\""
                
                _driver._nu_index = _driver._nu_index + 1
                _response = _driver.sendCommand(_command)
                _new = True
                
            print(_command, _response)

            if "ERROR" in _response:
                _driver._sendMessage(_driver._admin, "Error!" + _response)
                return
            
            # add the newly created admin to the user list
            if _new :
                _driver._user_list[new_admin_no] = Driver.User(_driver._nu_index - 1, new_admin_no, "ADMIN," + str(_costPerMinute) )

            # change old admin
            old_adm_index = str(_driver._user_list[_driver._admin]._pb_index)
            _command = "AT+CPBW=" + old_adm_index + ", \"" +  _driver._admin + "\", 129, \"0\" "
            _response = _driver.sendCommand(_command)

            print(_command, _response)

            if "ERROR" in _response:
                _driver._sendMessage(_driver._admin, "Error!" + _response)
                return
    
            # unset the admin flag for prev admin
            _driver._user_list[_driver._admin].balance = 0
            _driver._user_list[_driver._admin]._is_admin = False
            _driver._user_list[_driver._admin]._rate = 0

            #set the admin flags for new incoming admin
            _driver._admin = new_admin_no
            _driver._user_list[_driver._admin]._is_admin = True
            _driver._user_list[_driver._admin]._rate = _costPerMinute
            _driver._sendMessage(_driver._admin, "Welcome New Admin!")


        # routine to change the rate per minute in admin record
        def _changePerMinRate(self, costPerMinute, _driver):
            print ("Incoming rate " + costPerMinute)
            if float(costPerMinute)==0: return
            _index = _driver._user_list[_driver._admin]._pb_index
            _command = "AT+CPBW=" + str(_index) + ", \"" + _driver._admin + "\", 129,\"ADMIN," + costPerMinute + "\""
            _response = _driver.sendCommand(_command)
            if 'OK' in _response:
                _driver._sendMessage(_driver._admin, "Rate changed to " + costPerMinute)
            else:
                _driver._sendMessage(_driver._admin, "Rate change FAILED! " + _response)
            
            _driver._user_list[_driver._admin]._rate = float(costPerMinute)
#             _driver._wdog.feed()
            utime.sleep(100)

    
        def _addUpdateUser(self, phone, balance, _driver, auto_update=False):
            print("Adding / Updating user..", balance, auto_update)
            
            _driver._await_exec()

            utime.sleep_ms(500)
            
            _book_write = lambda ind, ph, bal : "AT+CPBW={0},\"{1}\",129,\"{2}\"".format(ind, ph, bal)
            
            _new_balance = balance
            if phone in _driver._user_list:
                # update the existing user with new balance
                _pb_index = _driver._user_list[phone]._pb_index
                _new_balance = _driver._user_list[phone]._balance + _new_balance
                _command = _book_write(_pb_index, phone, _new_balance)
                print (_command)
#                 _command = "AT+CPBW=" + str(_pb_index) + ", \"" + phone + "\", 129,\"" + str(_new_balance) + "\""
                _response = _driver.sendCommand(_command);
                utime.sleep_ms(1000)
                print(_response)

                if not "OK" in _response:
                    print("Error ", _response)
                    _driver._sendMessage(_driver._admin,
                                         "Unable Update User{0}, {1} ".format(phone, _response))
                    
                    # if error occurs during water dispensing , abort the run
                    if auto_update:
                        _driver._motor._abort()
                    
                    return False
                if _new_balance < 50.0:
                        _driver._sendMessage(phone, "Balance less than 50")
                        return False
                
                _driver._user_list[phone]._balance = _new_balance

            else:
                # add new user with balance
#                 _command = "AT+CPBW=" + str(_driver._nu_index) + ", \"" + phone + "\", 129,\"" + str(_new_balance) + "\""
                _command = _book_write(_driver._nu_index, phone, _new_balance)
                _response = _driver.sendCommand(_command);

                if "ERROR" in _response:
                    _driver._sendMessage(_driver._admin,
                                         "Unable add User{0}, {1} ".format(phone, _response))

                    if auto_update:
                        _driver._motor._abort()
                    
                    return False
                
                _driver._nu_index = _driver._nu_index + 1
                _driver._user_list[phone] = Driver.User(_driver._nu_index - 1, phone, str(_new_balance))

            # dont send the message if the motor is running and the update is from the process
            if auto_update == True: return True
            
            print("sending messages...")
            # send message to admin and client of the new balance update
            _driver._sendMessage(phone, "New Balance is "  + str(_new_balance))
            _driver._sendMessage(_driver._admin, "New Balance for " + phone + " is "  + str(_new_balance))
            
            return True
                
    def _stop_motor(self):
        self._motor_running = False
        self._motorPower.value(0)
        

    def _start_motor(self):
        self._motor_running = True
        print("Motor Start ", self._motor_running)
        #power on the pump
        self._motorPower.value(1)
        
    
    def _is_motorRunning(self):
        return self._motor_running
    
    
    def _reset_modem(self):
        #code to reset the modem
        print("reset modem!")
    

    def _initialize_book(self):
        
        _is_alive = self.sendCommand("AT")

        if len(_is_alive)==0:
            print("GSM Not Responding!")
            return

        _cmd = "AT+CPBR=1,50"

        while True:
            _address_book = self.sendCommand(_cmd)
            
            if 'ERROR' in _address_book:
                utime.sleep_ms(1000)
                if not DEBUG: self._wdog.feed()
                continue
            
            if 'OK' in _address_book: break
        
        print(_address_book, _address_book.split('\n'))
        
        if len(_address_book.split('\n')[1])=='':
            #reset modem
            self._reset_modem()
            return
        
        if "ERROR" in _address_book:
            print("ERROR in Initialization!")
            return False
        
        _pb_index = 0
        self._user_list.clear()
        
        for _user in _address_book.split('\n')[1:-1]:
            if not DEBUG: self._wdog.feed() # watch dog 
            user_obj = _user.split(',')
            if user_obj[0]=='OK': continue
            _pb_index = int(user_obj[0][6:].strip())
            _phone = user_obj[1]
            _balance = user_obj[3]

            self._user_list[_phone] = self.User(_pb_index, _phone, _balance)
            
            if 'ADMIN' in _balance:
                self._admin = _phone
                self._user_list[self._admin]._rate = float(user_obj[4])
        
        self._nu_index = _pb_index + 1
        
        #initialize the motor to default
        self._motor = Motor(self._admin, self)
        
        return True


    def _sendMessage(self, phone, text_message):
        if text_message=='' : return

        self._await_exec()
        self._in_progress = True

        print("AT+CMGS=\"" + phone + "\"", text_message)
        self.sendCommand("AT+CMGS=\"" + phone + "\"", True)
        self.sendCommand(text_message, True)
        self.sendCommand('\x1a', True)
        if not DEBUG: self._wdog.feed() # watch dog
        utime.sleep_ms(500)
        
        _retry = 0

        print("Awaiting OK response...")
#         self._in_progress = True
        while True:
            if not DEBUG: self._wdog.feed() # watch dog 
            _response = self._getResponse()
            print(_retry, sep = ' ')
            if 'OK' in _response or _retry > 20:
                break
            _retry = _retry + 1
            
            utime.sleep_ms(500)
        
        self._in_progress = False
        

    # check whether the GSM is being accessing by other processes
    def _await_exec(self, consumer=''):
        while self._in_progress:
            if not DEBUG: self._wdog.feed()
            utime.sleep_ms(400)
            print("Awaitng...", consumer)
        return True


    def _await_que(self):
        while self._que_locked:
            utime.sleep_ms(10)
        self._que_locked = True


    def _verify_request(self, _request):
        print("verifying.....") 
        _user_data = _request.split(',')
        
        # do not dispense water for ADMIN
        if _user_data[4]=='ADMIN':
            # send usage statistics by message 
            self._sendMessage(self._admin, " Working in Good Health")

            self._await_que()
            del self._requestQue[0]
            self._que_locked = False
            return
        

        print(_user_data)
        # this find the appropriate number particularly in case of land line numbers
        _phone = _user_data[0][10:]
        if _phone not in self._user_list.keys():
            for item in self._user_list.keys():
                if _phone in item:
                    _t_phone = self._user_list[item]._phone
                    break
        
        try:
            _balance = 0.0 if 'ADMIN' in _user_data[4] else float(_user_data[4])
        except ValueError as er:
            print ("value Error", er)
            self._await_que()
            del self._requestQue[0]
            self._que_locked = False
            return
        
        print("Serving {0}, and requesting {1}, running {2}".format(self._motor._serving_user(),
                                                                   _phone,
                                                                   self._is_motorRunning()))
    
        # insuffient balance , send message and abort
        if  _balance<50.0:
            self._sendMessage(_phone, "Low Balance {:4.2f} ".format(_balance))
            print("Low Balance ..")
            print("Queueu status ", self._requestQue)

            if _phone in self._requestQue[0]:
                a = self._await_que()
                del self._requestQue[0]
                self._que_locked = False
            

        # if user calling more than once means cancellation request
        elif _phone in self._requestQue:
            test_duplicate = list(filter(lambda x: x>0, [ index+1 if _phone==item else 0
                                                for index, item in enumerate(self._requestQue)]))
            if len(test_duplicate)>1:
                print(_phone + " Cancelled !\n");
                self._sendMessage(_t_phone, " Request Cancelled! ")

                self._await_que()
                del self._requestQue[test_duplicate[::-1][0] - 1 ]
                self._que_locked = False

        elif self._motor._serving_user() == _phone and self._is_motorRunning():

            # request to calloff the water dispense
            self._await_que()
            del self._requestQue[0]
            self._que_locked = False

            self._stop_motor()
#             self._sendMessage(_t_phone, "Stopped Water")
            
        else:
            # if the motor is running for other user
            if self._is_motorRunning():
                token = 1
                for message in self._requestQue:
                    if '+CLIP' in message :
                        if _phone not in message:
                            token = token + 1
                        else :
                            break
                
                print("{} Request Pending , please wait ".format(token))
                self._sendMessage(_t_phone, " {}  Request Pending , please wait ".format(token))
                return

            print("Before :Total pending list ", self._requestQue)
            #remove the top most request from que
            if _phone in self._requestQue[0]:
                self._await_que()
                del self._requestQue[0]
                self._que_locked = False
            
            print("After:Total pending list ", self._requestQue)
            
            self._make_check_call(_t_phone)
            
            # dispense Water thru motor class
            self._motor = Motor(_phone, self)
#             self._motor_timer.init(mode=Timer.ONE_SHOT, callback=_motor._dispenseWater, period=100)
            _thread.start_new_thread(self._motor._dispenseWater,())
            print ("Successfuly started motor...", self._is_motorRunning())



    # CALL THE NUMBER FOR SECONDARY CONFIRMATION
    def _make_check_call(self, phone):
        
        while True:
            _response = self.sendCommand("AT")
            if 'OK' in _response: break
            utime.sleep(1000)
        
#         utime.sleep_ms(1000)            
        
        _response = self.sendCommand("ATD{};".format(phone))
        print("ATD{};".format(phone))
        if DEBUG: print(_response)
        
        if 'ERROR' in _response:
            self.sendCommand("ATH")
            return
            
        utime.sleep_ms(2000)
            
        while True:
            _response = self.sendCommand("AT+MORING=1")
            print(_response)
            if 'MO RING' in _response: break
            if 'NO CARRIER' in _response: break
            
            if 'ERROR' in _response: break
            utime.sleep_ms(200)

        utime.sleep_ms(2000)
        self.sendCommand("ATH")


    def _processInCoText(self, response):
        
        if len(response)==0:
            return
        
        if 'ERROR' in response:
            self._sendMessage(self._admin, " Error in _Processing Text Messages ->" + response)
            return
        
        print(response)
        
        _frame = response.splitlines()
        _sender_info = _frame[1].split(',')
        _message = _frame[2].upper().split(' ')
        
        _in_phone = _sender_info[0][9:].strip()
        print("processing for ..." + _in_phone )
        print(_message)
        
        #capture all portal messages from the service provider
        if 'BSPRTL' in response :
            self._sendMessage(self._admin, response)
            return 
        
        _exec = _message[0]
        _is_admin =  True if _in_phone == self._admin else False
#         self._checkAdmin(_sender_info[0], _in_phone)
        print("Is Admin " , _is_admin, self._admin, _in_phone)
        
        if _is_admin:
            
            print("procesing admin functions...")
                  
            if len(_in_phone)!=10:
                self._sendMessage(self._admin, " Invalid Phone Number!" + _in_phone)
                return
                
            if 'ADD' in _message:
                if len(_message)!=3:
                    self._sendMessage(self._admin, " Wrong format for ADD!")
                    return

                _phone, _balance = _message[1], _message[2]
                index = self._user_list[self._admin]._pb_index
                _admin = Driver.Admin(index, self._admin)
                _admin._addUpdateUser(_phone, float(_balance), self)
                del _admin
                return
            
            # messages with three parameters in the message

            if 'RATE' in _message:
                if len(_message)==1:
                    self._sendMessage(_in_phone,
                                      "The Current per Min Rate is {:4.2f}".format(self._user_list[self._admin]._rate))
                    return

                if len(_message)!=2:
                    self._sendMessage(self._admin, " Wrong format for RATE! ")
                    return
                
                _new_rate = _message[1]
                
                index = self._user_list[self._admin]._pb_index
                _admin = Driver.Admin(index, self._admin)
                
                _admin._changePerMinRate(_new_rate, self)
                self._sendMessage(_in_phone,
                                  "The Current per Min Rate is {:4.2f}".format(self._user_list[self._admin]._rate))
                del _admin

            elif 'CHANGE' in _message:
                if len(_message)!=2:
                    self._sendMessage(self._admin, " Wrong format for CHANGE! ")
                    return
                
                # the message format should be 'CHANGE #PhoneNumber'
                index = self._user_list[self._admin]._pb_index
                
                old_admin = Driver.Admin(index, self._admin)
                new_admin_no = _message[1]
                print("Changing admin  to " + new_admin_no);
                old_admin._changeAdmin(new_admin_no, self)
                del old_admin
            
            elif 'BAL' in _message:
                
                # get the balance of the current users
                print("balance list", _message)
                if len(_message)==1:
                    _bal_list = ""
                    
                    for phone in self._user_list:
                        if self._user_list[phone]._is_admin: continue
                        _bal_list = _bal_list + phone + " : " + str(self._user_list[phone]._balance) + EOL
                    
                    print(_bal_list)
                    self._sendMessage(self._admin, _bal_list)
                    return
                
                if len(_message)!=2:
                    self._sendMessage(self._admin, " Wrong format for BAL! ")
                    return

                _phone_no = _message[1]
                _cmd_string = " The balance available for " + _phone_no  + " is " + \
                              str(self._user_list[_phone_no]._balance )
                print(_cmd_string)
                self._sendMessage(self._admin, _cmd_string)

        elif _exec.upper().startswith("BAL"):

            self._sendMessage(_in_phone,
                              "Balance is {:4.2f}".format(self._user_list[_in_phone]._balance))


    def _checkAdmin(self, phone, adm_message):
        if not self._admin in phone:
#             print ( phone)
#             print("Sending UNAUTH message..." )
            self._sendMessage(self._admin, "Unautharized use of Admin func! " + adm_message)
            return False
        return True


    def _processQue(self, timer):
        
#         self._wdog.feed()

        track_index = 0
#         print("Processing Request Que...")
        while len(self._requestQue) > 0:

            self._await_exec("Process Que");
            
            _response = self._requestQue[track_index]
            print(_response)
 
            if '+CLIP:' in _response:
                self._verify_request(_response.splitlines()[2])
                
            elif '+CMT:' in _response:
                while (self._que_locked):
                    utime.sleep_ms(10)

                self._que_locked = True
                del self._requestQue[track_index]
                self._que_locked = False

                self._processInCoText(_response)
            
            utime.sleep_ms(1000)
#             if not DEBUG: self._wdog.feed()

            track_index = track_index + 1
            track_index = 0 if track_index == len(self._requestQue) else track_index


    def sendCommand(self, _command, cont_lock=False):
        
        if cont_lock==False:
            self._await_exec('SendCommand :' + _command)
        
        self._in_progress = True

        if (_command == '' or len(_command)==0): return 
        _command = _command + EOL

        for it in _command:
            self._led.toggle()
            self._gsm.write(it)
            utime.sleep(0.1)
            if not DEBUG: self._wdog.feed() # watch dog 
    
        utime.sleep(0.2)
        self._led.value(0)
        _reponse = self._getResponse()
        self._in_progress = cont_lock
        return _reponse


    def _getResponse(self):
        
        _response =''
        _cmd_string = True
        ch = ''
        
        while self._gsm.any():
            ch = self._gsm.read(1)
            if ch in PASS_THRU_STRS or ch.isalpha() or ch.isdigit() :
                _response = _response + ch.decode('utf-8')
            self._led.toggle()
            utime.sleep(0.01)
            if not DEBUG: self._wdog.feed() # watch dog 

        self._led.value(0)
        return _response.replace('\n\n', '\n')

    
    def _is_modemActive(self):
    
        while True:
            response = self.sendCommand("AT")
            if 'OK' in response: break
            utime.sleep_ms(1000)
    
    
    def receive_incoming(self):

        self._wdog = WDT(timeout=3000)

        self._is_modemActive()

        self._initialize_book()
        utime.sleep_ms(500)

        self.processRequest = Timer()
        
        self.processRequest.init(freq=.5, mode=Timer.PERIODIC, callback=_driver._processQue)
        
        _loop = True

        while _loop:
#             print("Waiting requests...", self._in_progress)

            try:

                self._await_exec('Waiting requests')
                self._in_progress = True
                
                if self._gsm.any():
                    
                    utime.sleep_ms(50)
                    _response = self._getResponse()
                    self._in_progress = False
                    
                    print ("Incoming request..->", _response)                    
                    if any([True if cmd in _response else False for cmd in ACCEPTED_COMMANDS]):
                        
                        print(_response, _response.split('\n'))
                        if '+CLIP' in _response:
                            self.sendCommand("ATH")
                        
                        while self._que_locked:
                            utime.sleep_ms(10)
                        
                        self._que_locked = True
                        self._requestQue.append(_response)
                        self._que_locked = False
                
                self._in_progress = False
                utime.sleep_ms(1000)
                if not DEBUG: self._wdog.feed()
            
            except KeyboardInterrupt as err:
                _loop = False

        print("exiting...")
        self.processRequest.deinit()
    

class Motor:
    
    def __init__(self, phone, driver):
        self._counter = 0
        self._start_time = 0;
        self._phone = phone
        self._driver = driver
        self._motor_runner = Timer()
        self._break_loop = False
        self._duration = 5 # in minutes
    
    
    def _del__(self):
        self._break_loop = True


    def _serving_user(self):
        return self._phone
    

    def _abort(self):
        self._mot_running = False
    

    def _dispenseWater(self):
        self._lap_end_time = utime.ticks_add(utime.ticks_ms(), 1*60*1000)
        
        _admin = Driver.Admin(self._driver._user_list[self._driver._admin]._pb_index, self._driver._admin)
        _admin._addUpdateUser(self._phone,
                              -1 * self._driver._user_list[self._driver._admin]._rate * 1,
                              self._driver, True)
        print("Start the motor process...", self._driver._is_motorRunning())
        self._driver._start_motor()
        self._counter = 1

        total_bill = self._driver._user_list[self._driver._admin]._rate
        while self._driver._is_motorRunning():
#             print("Running..", utime.ticks_diff( self._lap_end_time, utime.ticks_ms()))
            if utime.ticks_diff( self._lap_end_time, utime.ticks_ms()) <= 0:
                # block to charge the account for next cycle
                self._counter = self._counter + 1
                # break when the time exceeded _duration variable
                if self._counter > self._duration or self._break_loop : break
                self._lap_end_time = utime.ticks_add(utime.ticks_ms(), 60*1000)
                total_bill = total_bill + self._driver._user_list[self._driver._admin]._rate

                _success = _admin._addUpdateUser(self._phone,
                                               -1 * self._driver._user_list[self._driver._admin]._rate * 1,
                                               self._driver, True)
                if not _success : break
                print("Completed {0} cycles".format(self._counter), self._driver._is_motorRunning())
            utime.sleep_ms(1000)
        
        self._driver._stop_motor()

        print("Total Bill {:4.2f}".format(total_bill))
        self._driver._sendMessage(self._phone,
                                  "Total Bill {:4.2f} for {:3d} Minutes".format(total_bill, self._counter))


_driver = Driver(gsm, led)

if DEBUG:
    _driver._initialize_book()
else:
    _driver.receive_incoming()
    del _driver

