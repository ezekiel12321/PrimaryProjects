
from collections import deque # Used to implement queues.
import random # Random choice, etc.
import heapq # Used in discrete event simulator
import numpy as np # Used for gamma probability distribution, and percentiles.
import matplotlib.pyplot as plt
import itertools
from tabulate import tabulate # To display the bus status.

def fmt(x):
    """Formats a number x which can be None, for convenience."""
    return None if x is None else "{:.2f}".format(x)

class Event(object):

    def __init__(self, method, delay=0, args=None, kwargs=None):
        """An event consists in calling a specified method after a delay,
        with a given list of args and kwargs."""
        self.method = method
        self.delay = delay
        self.args = args or []
        self.kwargs = kwargs or {}
        self.time = None # Not known until this is added to the queue.

    def __call__(self, time):
        """This causes the event to happen, returning a list of new events
        as a result. Informs the object of the time of occurrence."""
        return self.method(*self.args, time=time, **self.kwargs)

    def __lt__(self, other):
        return self.time < other.time

    def __repr__(self):
        return "@{}: {} {} {} {} dt={:.2f}".format(
            fmt(self.time),
            self.method.__self__.__class__.__name__,
            self.method.__name__,
            self.args, self.kwargs, self.delay
        )

class EventSimulator(object):

    def __init__(self, trace=False):
        self.events = []
        self.time = 0 # This is the global time.
        self.trace = trace

    def add_event(self, event):
        """Adds an event to the queue."""
        event.time = self.time + event.delay
        heapq.heappush(self.events, event)

    def step(self):
        """Performs a step of the simulation."""
        if len(self.events) > 0:
            event = heapq.heappop(self.events)
            self.time = event.time
            new_events = event(self.time) or []
            for e in new_events:
                self.add_event(e)
            if self.trace:
                print("Processing:", event)
                print("New events:", new_events)
                print("Future events:", self.events)

    def steps(self, number=None):
        """Performs at most number steps (or infinity, if number is None)
        in the simulation."""
        num_done = 0
        while len(self.events) > 0:
            self.step()
            num_done += 1
            if num_done == number:
                break

"""## The Classes / Actors in the Simulation

### Person

Here is the class for a person.  We give it to you as it contains really nothing very interesting, and having named properties for the transit times of a person helps us in writing the visualization code later on.  

Notice how every person has a source and a destination.  You have to make sure that people get off the bus at the right stop!
"""

class Person(object):

    def __init__(self, start_time, source, destination, have_arrived,
                 person_id=None):
        """
        @param start_time: time at which a person enters the system.
        @param source: stop at which the person wants to climb on.
        @param destination: destination stop.
        @param have_arrived: list of people who have arrived, so we can
            plot their bus time.
        """
        self.start_time = start_time
        self.bus_time = None # Time at which climbed on bus
        self.end_time = None
        self.source = source
        self.destination = destination
        self.have_arrived = have_arrived
        # for id purpose
        self.id = person_id

    # Event method
    def arrived(self, time=None):
        """The person has arrived to their destination."""
        self.end_time = time
        self.have_arrived.append(self)
        return [] # No events generated as a consequence.

    def start_bus(self, time=None):
        """The person starts getting on the bus."""
        self.bus_time = time

    @property
    def elapsed_time(self):
        return None if self.end_time is None else self.end_time - self.start_time

    @property
    def travel_time(self):
        return None if self.end_time is None else self.end_time - self.bus_time

    @property
    def wait_time(self):
        return None if self.end_time is None else self.bus_time - self.start_time

    def __repr__(self):
        return f"Person #: {self.id}, source: {self.source}, dest: {self.destination}"

"""### Source

Here is the `Source` class, in charge of generating people and adding them to the bus stops.  There isn't much interesting going on here, and we provide the implementation for you so the statistical process of people generation is clear. 
"""

class Source(object):
    """Creates people, and adds them to the queues."""

    def __init__(self, rate=1., queue_ring=None, number=None, have_arrived=None):
        """
        @param rate is the rate at which people are generated.
        @param number is the total number of people to generate; None = unlimited.
        @param queue_ring is the queue ring (a list of queues) where people are added.
        @param have_arrived is the list where people who have arrived are added.
        """
        self.rate = rate
        self.queue_ring = queue_ring
        self.num_stops = len(queue_ring)
        self.number = number
        self.have_arrived = have_arrived
        self.person_id = 0 # For debugging.

    # Event method
    def start(self, time=None):
        if self.number == 0:
            return [] # Nothing more to be done.
        # Creates the person
        self.person_id += 1
        source, destination = random.sample(range(self.num_stops), 2)
        person = Person(time, source, destination, self.have_arrived,
                        person_id = self.person_id)
        queue = self.queue_ring[source]
        enter_event = Event(queue.enter, args=[person])
        # Schedules the next person creation.
        self.number = None if self.number is None else self.number - 1
        dt = np.random.gamma(1, 1/self.rate)
        start_event = Event(self.start, delay=dt)
        return [enter_event, start_event]

"""### Question 1: Implement a Queue

Here is the class `Queue`.  Every bus stop has a queue.  The `Source` (code above) calls the method `enter` of the queue to add people to it.  When the bus stops at a given stop, the people in the queue need to climb on, if there is space for them in the bus.  

We leave it up to you to implement the queue.  Feel free to add any methods and properties you need. 
"""

### Class Queue

class Queue(object):

    '''
    This queue is used to add people only Ig, I have to check the amount of people on the bus
    I have to add these people one by one to the queue

    '''
    def __init__(self):
        """We create a queue."""
        self.people = deque()
   
    def returnPeoples(self):
        return self.people

    def popBop(self):
        if len(self.people) != 0:
            return self.people.popleft()
        return None

    # Event method
    def enter(self, person, time=None):
        
        self.people.append(person)

    def __len__(self):
        print("Running Queue len method", len(self.people))
        return len(self.people)
    @property
    def empty(self):
        return len(self.people) == 0
        #check to see whether bus is at max
        #is there where I have the people get on the bus? What do I do here?
        
    
    
        # YOUR CODE HERE

    ### You can put here any other methods that might help you.
    # YOUR CODE HERE

"""### Question 2: Implement a Bus

The bus is the centerpiece of our simulation.  The bus behaves as follows. 

When the bus arrives to a stop: 

* first people for whom the stop is the destination get off, one by one, 
* then people waiting get on, one by one, up to when the queue is empty or the max capacity of the bus is reached,
* and finally, it drives to the next station. 

A person climbing on or off the bus takes time given by `np.random.gamma(2, geton_time / 2)`.  You are advised to model as events the climbing up/down of every person. 

The bus takes time given by `np.random.gamma(10, nextstop_time / 10)` to drive to the next stop. 

The initializer of the bus takes as parameters the ring of queues (one queue for each stop), so that it has access to the people climbing on/off the bus.   It also takes as parameters the bus max capacity, and the times for people to get on and off. 

We leave it up to you to write all the interesting bits of the bus implementation.
"""

#@ title Class Bus

class Bus(object):

    def __init__(self, queue_ring, max_capacity, geton_time, nextstop_time,
                 bus_id=None):
        """The bus is created with the following parameters:
        @param max_capacity: the max capacity of the bus.
        @param queue_ring: the ring (list) of queues representing the stops.
        @param geton_time: the expected time that it takes for a person to climb
            the 2 steps to get on the bus.  The time a person takes to get on is
            given by np.random.gamma(2, geton_time / 2).
            This is the same as the time to get off the bus.
        @param nextstop_time: the average time the bus takes to go from one stop
            to the next.  The actual time is given by
            np.random.gamma(10, nextstop_time/10).
        @param bus_id: An id for the bus, for debugging.
        """
        self.queue_ring = queue_ring
        self.max_capacity = max_capacity
        self.geton_time = geton_time
        self.nextstop_time = nextstop_time
        self.id = bus_id
         
        ### Put any other thing you need in the initialization below.
        # Maybe a list of people or a queue (not sure)
        self.current_stop=0
        self.people_on_bus = []
        self.tempQueue = deque()
        self.count=None
        self.list_of_people_getting_off = []
        self.couldNotFindAnyone = False
        self.listOfJustAdded = []
        
        



    @property
    def stop(self):
        """Returns the current (most recent) stop of the bus,
        as an integer."""

        #Enumeration
        #print(self.count)
        return self.count
        """
        Length is 7:
        0 1 2 3 4 5 6 7 8 9 10 
        """ 

        # YOUR CODE HERE

    @property
    def onboard(self):
        """Returns the list of people on the bus."""
        #Returns the list version of the queue or just list?
        
        # YOUR CODE HERE
        return self.people_on_bus

    @property
    def occupancy(self):
        """Returns the number of passengers on the bus."""
        
        return len(self.people_on_bus)
    # Event method.
    def arrive(self, stop_idx, time=None):
        '''
        first people for whom the stop is the destination get off, one by one,
        then people waiting get on, one by one, up to when the queue is empty or the max capacity of the bus is reached,
        and finally, it drives to the next station.
        '''
        #This needs to be mutable otherwise im screwed
        self.count = stop_idx
        self.tempQueue = self.queue_ring[self.stop]
        
        #Are there people who are at their destination still on the bus?
        #print("Could not find anyone",self.couldNotFindAnyone)
        #print("Numbner of people on bus",len(self.people_on_bus))
        #print("People at destination to drop off",self.anyone_at_destination())
        #return [Event(self.get_off_bus, delay=0)]
        #It recounts itself after adding people to the bus
        #I need to find a way that so after it finishes adding
        
        
        if len(self.people_on_bus) > 0 and self.couldNotFindAnyone is False and self.anyone_at_destination() is True:
            dt = np.random.gamma(2, self.geton_time / 2)
            return [Event(self.get_off_bus, delay=dt)]
        
        #print("lenght of queue of people is",len(self.tempQueue.returnPeoples()))
        if len(self.tempQueue) > 0 and self.space_on_bus > 0:
            dt = np.random.gamma(2, self.geton_time / 2)
            return [Event(self.get_on_bus, delay=dt)]
        else:
            #print("Leaving")
            dt = np.random.gamma(10, self.nextstop_time/10)
            return [Event(self.driveAway,delay=dt)]
        
            

        #Event(self.get_off_bus, delay=np.random.gamma(2, 10))
        #Use the stop function to figure out where I am, Use a for loop for the people and find out who has reached their destination
        #
    

    
    def get_off_bus(self,time=None):
        #for loop make sure they are at their desination
        self.couldNotFindAnyone = False
        #print("top of get off bus")
        tempCount = 0
        for x in self.people_on_bus:
            if x.destination == self.count:
                guyKickedOff = self.people_on_bus.pop(tempCount)
                guyKickedOff.arrived(time=time)               
                return [Event(self.arrive,delay=0,args=[self.stop])]
            tempCount += 1
        
        self.couldNotFindAnyone = True
        
        print("should do this")
        return [Event(self.arrive,delay=0,args=[self.stop])]
    def get_on_bus(self,time=None):
        
        #return Event(arrive,delay=0)
        print("Error case:",len(self.tempQueue))
        guyComingOn = self.tempQueue.popBop()
        if guyComingOn != None:
            guyComingOn.start_bus(time=time)
            self.people_on_bus.append(guyComingOn)
        print("get on bus")
        return [Event(self.arrive,delay=0, args=[self.stop])] #Adds those people onto the bus
    def __repr__(self):
        """This will print a bus, which helps in debugging."""
        return "Bus#: {}, #people: {}, dest: {}".format(
            self.id, self.occupancy, [p.destination for p in self.onboard])

    ### You can have as many other methods as you like, including other
    ### events for the bus.  Up to you.
    # YOUR CODE HERE
    @property
    def space_on_bus(self):
        self.space = self.max_capacity - self.occupancy
        return self.space
    
    def anyone_at_destination(self):
        
        
        #print("self count",self.count)
        for x in self.people_on_bus:
            #print("x destination is",x.destination)
            if x.destination==self.count:
                
                #print("self count",self.count)
                return True
            
                
        return False
        
    
    def driveAway(self, time=None):
        print("driving away")
        self.couldNotFindAnyone = False
        self.count += 1
        self.count = self.count%len(self.queue_ring)
        print("self.count is", self.count)
        return [Event(self.arrive,delay=0, args=[self.count])]

"""## Simulation setup

Let us define some functions that will help in the visualization of the simulation results.
"""

def bus_distance(ix, iy, num_stops=20):
    """Returns the distance between two buses."""
    if ix is None or iy is None:
        return None
    d1 = (ix - iy + num_stops) % num_stops
    d2 = (iy - ix + num_stops) % num_stops
    return min(d1, d2)

"""This class sets up the simulation.  It is given to you for convenience, and so that you have a way to do the plots.  
Feel free to add methods to it, if you need them for debugging.

"""

class Simulation(object):

    def __init__(self, num_stops=20, num_buses=1,
                 bus_nextstop_time=1, bus_geton_time=0.1,
                 bus_max_capacity=50,
                 person_rate=2, destinations="random",
                 number_of_people=None,
                 trace=False):
        self.num_stops = num_stops
        self.num_buses = num_buses
        self.bus_max_capacity = bus_max_capacity
        # Chooses the initial stops for the buses.
        self.initial_stops = list(np.mod(np.arange(0, self.num_buses) * max(1, num_stops // num_buses), num_stops))
        # Speeds
        self.bus_nextstop_time = bus_nextstop_time
        self.bus_geton_time = bus_geton_time
        self.person_rate = person_rate
        # Event simulator
        self.simulator = EventSimulator(trace=trace)
        # Builds the queue ring
        self.queue_ring = [Queue() for _ in range(num_stops)]
        # And the source.
        self.have_arrived = []
        self.source = Source(rate=person_rate, queue_ring=self.queue_ring,
                             number=number_of_people, have_arrived=self.have_arrived)
        # And the buses.
        self.buses = [Bus(queue_ring=self.queue_ring,
                          max_capacity=bus_max_capacity,
                          geton_time=bus_geton_time,
                          nextstop_time=bus_nextstop_time,
                          bus_id=i + 1)
            for i in range(num_buses)]
        # We keep track of the distances between buses, and the
        # bus occupancies.
        self.positions = [[] for _ in range(num_buses)]
        self.occupancies = [[] for _ in range(num_buses)]


    def start(self):
        """Starts the simulation."""
        # Injects the initial events in the simulator.
        # Source.
        self.simulator.add_event(Event(self.source.start))
        # Buses.
        for i, bus in enumerate(self.buses):
            self.simulator.add_event(
                Event(bus.arrive, args=[self.initial_stops[i]]))

    def step(self):
        """Performs a step in the simulation."""
        self.simulator.step()
        for bus_idx in range(self.num_buses):
            self.positions[bus_idx].append(self.buses[bus_idx].stop)
            self.occupancies[bus_idx].append(self.buses[bus_idx].occupancy)

    def plot(self):
        """Plots the history of positions and occupancies."""
        # Plots positions.
        for bus_idx in range(self.num_buses):
            plt.plot(self.positions[bus_idx])
        plt.title("Positions")
        plt.show()
        # Plots occupancies.
        for bus_idx in range(self.num_buses):
            plt.plot(self.occupancies[bus_idx])
        plt.title("Occupancies")
        plt.show()
        # Plots times.
        plt.hist([p.wait_time for p in self.have_arrived])
        plt.title("Wait time")
        plt.show()
        plt.hist([p.travel_time for p in self.have_arrived])
        plt.title("Time on the bus")
        plt.show()
        plt.hist([p.elapsed_time for p in self.have_arrived])
        plt.title("Total time")
        plt.show()
        # Plots bus distances
        if self.num_buses > 1:
            for i, j in itertools.combinations(range(self.num_buses), 2):
                ds = [bus_distance(pi, pj, num_stops=self.num_stops)
                      for pi, pj in zip(self.positions[i], self.positions[j])]
                plt.plot(ds)
            plt.title("Bus distances")
            plt.show()

    def status(self):
        """Tabulates the bus location and queue status."""
        headers = ["Stop Index", "Queue", "Buses"]
        rows = []
        for stop_idx, queue in enumerate(self.queue_ring):
            buses = [b for b in self.buses if b.stop == stop_idx]
            busStr = "\n".join([bus.__str__() for bus in buses])
            personStr = "\n".join([person.__str__() for person in queue.people])
            row = [f"{stop_idx}", f"{personStr}", f"{busStr}"]
            rows.append(row)
        print(tabulate(rows, headers, tablefmt="grid", stralign='left', numalign='right'))

"""Here is how to use the above `status` mtethod to debug your application. """

simulation = Simulation(num_stops=5, num_buses=2, person_rate=2, trace=False)
simulation.start()
for i in range(30):
    simulation.step()
    print(f"\nState after step {i}")
    simulation.status()

"""## The bus goes round!

#### One bus

Let's start with one bus only.  This should produce plots similar to the ones given below.  You don't need to get _exactly_ the same, but if you get something wildly different, it might mean that you have made a mistake in your implementation.

![index4.png](data:image/png;base64,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)![index3.png](data:image/png;base64,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)![index2.png](data:image/png;base64,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)![index1.png](data:image/png;base64,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)![index.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAXAAAAEICAYAAABGaK+TAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO2dd3xd1ZXvv9tyl4vce6c4NtgYBMZAqAlJCCVMGskkpD5SHkwyybxMeJNMmEkyk8xjQpgZEuIUSCB0QjXNFFNMMLaxjYtcJFu2bFmWZFm2LFfp7vfHOUf36uqWU/ba8hHn9/noc65uWWvvffZeZ++1f3stpbUmQYIECRLED726uwAJEiRIkCAcEgOeIEGCBDFFYsATJEiQIKZIDHiCBAkSxBSJAU+QIEGCmCIx4AkSJEgQUyQGPMF7Ckqp/6uU+l2Bz/9WKfWCzTIlSBAWKuGBJzjRoZSqBsYA7UAr8Cxwo9b6YES5U4FtQB+tdVu0UiZIYB/JDDxBXHCV1noQcCZQDvygm8uTIEG3IzHgCWIFrfUunBn4aUqpq5VS65VSzUqpJUqp93nfU0r9o1Jql1KqRSm1SSl1mfv+LUqpe92vveZem5VSB5VSC5RSX1RKvZEh5zyl1HKl1H73el7GZ0uUUj9WSi119byglBrpftZfKXWvUmqvW77lSqkx0u2T4L2FxIAniBWUUpOAK4AW4H7g28Ao4BngKaVUX6XUqcCNwNla68HAh4DqHOIudK9lWutBWuu/ZukaDiwC/gsYAfwCWKSUGpHxtc8CXwJGA32Bf3Df/wIwFJjk/vbrwOHwNU+QoCsSA54gLnhcKdUMvAG8CmwAFmmtF2utjwO3AgOA83B85f2AWUqpPlrraq11VQidHwW2aK3v0Vq3aa3vBzYCV2V85y6t9Wat9WHgIeAM9/3jOIb7JK11u9Z6pdb6QIgyJEiQF4kBTxAXfExrXaa1nqK1/iYwHtjufai1TgE1wAStdSXOzPwWoF4p9YBSanwInZ10uNgOTMj4vy7j9SFgkPv6HuB54AGlVK1S6j+UUn1ClCFBgrxIDHiCuKIWmOL9o5RSOO6KXQBa6/u01he439HAz3PIKEbB6qTDxWRPRyForY9rrf9Faz0LZ1VwJXB9sd8lSBAEiQFPEFc8BHxUKXWZO7P9LnAUeFMpdapS6lKlVD/gCI7vOZVDRoP7/vQ8Op4BTlFKfVYp1Vsp9WlgFvB0scIppS5RSp2ulCoBDuC4VHKVIUGC0EgMeIJYQmu9Cfgc8N9AI45f+iqt9TEc//fP3PfrcDYYb84h4xDwU2CpyxQ5N+vzvTgz5+8Ce4HvAVdqrRt9FHEs8AiO8a7A8dvfE7ymCRLkR3KQJ0GCBAliimQGniBBggQxRWLAEyRIkCCmSAx4ggQJEsQUiQFPkCBBgpiit01lI0eO1FOnTrWpMkGCBAlij5UrVzZqrUdlv2/VgE+dOpUVK1bYVJkgQYIEsYdSKvtEMJC4UBIkSJAgtkgMeIIECRLEFIkBT5AgQYKYIjHgCRIkSBBTJAY8QYIECWKKxIAnSJAgQUyRGPAECRIkiCms8sAlsbXhII+vruX8GSOYP31E8R/4gNaaP75ZTVPrMd5/yijOnjrciNxsNLQc5f63d9DW7oSLPnfGCM6bMdK4ntrmwzy0ooZUSvOR08fxvnFDjMk+1pbirqXbaD3axoIZI1kww8w98NB48Cj3LXPaqF+fEr5w3lQG9TPXfbXW3PPWdhpbjjJlRCkfP2uiMdmvbW5gRXUTAJfPHstpE4ZGlrnnwBEeeLsGjeaT5ZOYUDYgssxM/OWdnVQ3tjJkQB++dP40Snopo/IBXtlYz6od++jXp4TrF0xhcH/zCYsy237S8IF8snyScR3bGlt5bNUu0Jor5oxj5lhz46oYeowBv+et7dy1tJqllY08+o3ziv/AB3buO8wtT20A4K1tTTz0tQVG5Gbj6Xdr+cXizR3/v7q5gSduvMC4nkdW7uSXL24BoGbfYW779BlFfuEfa3c18+/PbgTg9cpGHvvm+cZkAzy7dnenNpoxqpQPnzbOmPza/Uf45yfWA6AUXHPGeHqXmFmg/mTRBjbvOQhAVWMrd3z2zMgyn1i9i9tedNqjl1L83WUnR5bpoa09xXcfXoMXafqCk0eKGKVbnlrP9r2HAJg2spQrTjd3Pz1ktj3AVXPH079PiVEd9761nd+/sQ2Anc2H+cWnzI2rYugxLpS2dqe3tafMxTc/3p5OoJIyKDcbXtnX3nI5H3jfaNqFYrR7M/xpI0uNthPAcbcOfXv3EmkrT773EG03nNvGa5vpI0vRuniutWCyNVfOGcepYwYbaxuvPcBsnwdo1xqtYe6kMhH5HtratRUdV84Zx/c/MhMAiaHV1p6ibGAfpowYKGoncqHHGPCUwJ2xdS+8svdS5pepnfWAwErYle3VIa7ynavELUhpbfzeSiZi8URLtXVajxbXIdH2XXXIj9186DEGXKY727HgnhbpTqAR7Mwdg162DkpIvhZ8iGrMG0PJRFra0r20Yfgk2r6rDvkHUT70HAPeA2bg0g/xlJbT0TGDlREv3kbSM3DTDx7JvtnR1nIqOvRY0WHhQSTfWrnRYwx4SiDft4RbJhe0oPHIhGRnlnYDpYRnhZIz8FTK/L2V7Jt2XXrCOgTaPhs2XEH50HMMuNvpTHbrzIeCpCn3Nj68ziw1NnWGD9y0iswZskTxs33g2rCW9Azc/D3QGX5YU3IzV5zm76Vz9Qyf3LNCp3VIacjygZvuN+DYiY77a1x6YfQYAy7RcBI3O7ceB+L+QMENnY46CE1FpP2y3r2WKL6ID1xAZifh9IwZeOIDjwkklpSWPChZPke5npDSjnQJDZIuiEz5Yj5wd7Ul4kLRGoUyWnYr7jBh65DS2ooOhRL1UDt7S7I68qHHGHAJY2vLBy65gdZZj9wMPG0ARcSL+8AlaYopbd4Y2qGESq8ILc3ybTwkkhl4NEiwUGzNwHE7gPRuuRZkoXhNJUfzc65xpClqbV6uhMwO2e5Vnr0hzxCRbKe0koQHHhkStCqbM3AbHUBrLeajtnWQR45GKOgDF2ApSDIfpO+lBy24ikjrsHVYSFZHPvQgA+4aW4NGN/OhIHnyLZsPK6XK84E7Oswq6fBRo0TKn5bv/W9WfhcWisENbM8Pa1JupkzTjdFBazUqNZceLdYfPXRqJ6RcrTLsJT/oMQZc0mhIw9YMXNQHLnz8WtLn68gX9oEb54HH3wduhQduYZbfnT7wHhONUILyl/YFGhedpSfdASR1adyZgoibwLmK0QjdMABiPnxBH7sW8PV6G4AS7SF9Lz2kBF16Hry2lx5XvZRCd4MRLzoDV0r9QSlVr5Ral/HecKXUYqXUFvc6TLaYxSFyEjNlZyZiYzfe0WPDbyo3w5dsI9FYKAJll5z1WfOB29Bhg2t+gvvA7wY+nPXe94GXtNYnAy+5/3crJKMRisdrSNlZgkkeK7ayySjqQnGucrFQzMqUmNWnZTtXaWazzvJPS8CGeyOVkm+rfCjqQtFav6aUmpr19jXAxe7rPwJLgH80WK5AeHHDHjbvaTEm73h7ioWvbaVi9wFA7gn+0IoaqhoOsmxbk+gsoW7/Ee55q5o1O5uN69nXeoy7lm5jfa1cW63cvo+3qvaKzHKeXFPL+tr97G4+Apgt/6J3d/PurmaOtqWMlb31aBu/e30bK7bvM9oeWmt+/8Y2Gg4e5cDhNkCOP/3Ojn08v76OtpTcQZ43tjTyemWD0bbPRHtK85vXqth/+Dib9rTQr3f3bCeG9YGP0Vrvdl/XAWPyfVEpdQNwA8DkyZNDqiuMf3umgh1Nh4zJq9h9gP/3/Cb6lChGlPZlyADzqZ601vzjo+/SSyl691LMm1xmXIeHp9+t5Y5XqujbuxcXnTKKqoaDxX/kE0s21/NfL1fSt6QX44f2Z+Sgvhw5btaf9esllazZuZ+zp5r31P3oiXXsP3ycPiW9GFHal4nDzKUmu+Wp9ew9eJR+vUs4dewQXt/SGFnm29VN3PbiZvqW9GL+9OG8URldJjgZiX6yqILevRQlvRRD+vdm2shSwDxB4NdLqli8YQ8D+5ZwypjBPLO2zqwC4NYXNrFmZzP93bZvaj1qVP7mPS38x3OOjeilFFfNHc/K7fuM6vCDyJuYWmutlMp7i7XWC4GFAOXl5SK0jraU5uq542k+fJym1mNG5AEsvL6cS04dzed/v4yDR9siy82E1s7fTZedxLc/cErnz4xqSmc7WfXDD1LarzeX/ucSYzq8bEIvffciJg0fyJfvXs7h40cMSXd1pDRzJw7l4a+fR2W9s9IyVv6U5voFU7nl6tkA3PFKpSPfgIL2lOaz8yfzk4+dDsDvXt8aWW67296PfGMBcyaWMe3mRUbawpP7s4/P4RNuPtAXN+wxIDmHrpTmtAlDePqm91PVcLAjzZ9pHRefMoq7vnQOAAtfqwLM9RtvTN3x2TO5fPZYAC651dy48ouw8/49SqlxAO613lyRgsOjx5laKUnH9QB7VC1HF2K6bDAWMnm2piG5ySVB25TqN7Y2Lj1d8vRBWR02x28hhDXgTwJfcF9/AXjCTHHCwfRx2TSlzJjIrjrIrUMk0BSdNxhN6siO4idS/k67/KbpeJ0ZBCbHY/bDwUQfzaa2Gpu0uNfO5TUkPFtXxniVGmLZNsH0JmN64pJ+rztMuR8a4f3AX4FTlVI7lVJfAX4GfFAptQX4gPt/t8E0jUc6cJKjwzOqNuiDzlUm0p5zldyFl50ly60epNgnIDcDt8KG6gHH222O30Lww0L5TJ6PLjNcltAwnSrMRkqpDqqWFfqg3OC0sfSW5jyLzgINS5eiO2rLEwr5Y/qyY8sWxbgYesRRetP+LhtP157iA8+OISIByah1ktHqJGaBUv2mkNvQfPaj+PvAbeyT+UGPMOAdR8QxRHnK0ZlNU6kKDhjTwYlyZZsxpSLH0luirbLbyVQbSWZT0TncM1FLnavfmGgKG25DD7lcYsb7DLnrYq7fOOhaD7s8lJ5hwLXZeNq2ZpVg5wSXZH1sDHwJV4QH0+63zrLNu2fSJ46FfOBGpRbQZcEHLnpyV9AtGQQ9woCbjjiW7deVNHzZomWCE3X2IZusT9e2Mia6kw5vt9+0/OyltskHRVcmhBmZYL5v5nIbSrJQJMdWWodcXXKO324w5j3EgMfPB27Th2YjjKzkSkKy/IkP3IEN6qwHez5wOfmJD9wgTNPM8nG0TcLmgMnlQzYn233YCfakzD0Oo3KFGTT5/LCRZAq5rCSppl10WdAjHYUwnw/cNnqEATcdidAGpcomjzQl6EO2MfAzMwmZlguS/nXz9EcpvrZtHrh4hEBBeqgnHxIfuBGYftpKZ1gH+Qw2mcj2gZuEHc68zHJYOg+mhHtG6vxAIdeMaWKFZGgED5KuMbA7fguhRxjwTH+XCc5qrs5smhxk8+BELp+judyMzrVTWxkf8DI+0477nGMURq1DPhdZVLm5+2b0Bu/gtliaUHRpFxGueQ7dBuVD1/Frl0TYQwy4x7c11feyO7NEn87nQ5NYzmf7wE1qyI6zItFaUrEzcs1mTRmwXPfXhOxsucb6fI4Hg6TbzXT5C+mQUeBcOreXffQIA27a35WZYV0Kdn2OsnQtsOADF3FzOFeJ+yzlWpLzgSMiN7cuWf90hw5RF6g93nwhxD6p8W9f28rx9pQRA9V6tI07XqlMZ+IRerxtbTjIwte2OjqEesBf3tnJuzv3A7BsW5Pxzvzoyp2s3bWfNTubAZmBv2rHPp5YXUvd/sNMHm4u0QLA/kPH+e+XnTjUpu/BkePt3P6SK9ug8MUb9vDUmlrAfHs/s9bJzyL5IH5yTS3vbN/H7v1HGDfU7P0EZ+J156tb2XPgCAeOHBeZtOw/fJw7X61ic50Tl767WSixNuDNh47x02cqGNi3hNnjh7AlYlq1Fdv38aslVZT2LWHy8IGMHdLfUEk749F3dvLA8hpGDurHSaMHi+j46aIKWo600b+P8xQqn2I2m81PFm2g9Wg7/fv04oxJZfQWeBLdtbSap96tZXC/3sydZDZj0dKqRn73xjaGDujDzHFDjMpeXdPMr5dUMbh/b943ztz9vW3xZjbvaeGUMYMYajhL1GI3ecPk4QONys3Ez56poOHgUQb0KeGMSUONy29oOcrPn9tI/z696Ne7hDkTzet4a+tefu3aiInDBjC+TMZG+EWsDbiXFeMfPzyTq+aO5/FVuyLJ847H3vvV+cybbD59l4f2FPQt6cWKH3xATofWfOacSfzLNafJyHezzXiZbER0aM30kaW89N2Lzct27/UjX1/AyWPMPkS9fvTb68s5d/oIc3K15rL3jeY3ny83JtNDL6W4dt4EJuUw4KY25tq15uNnTuRnH59jSGJX+QC3XDWb686RSd/o3dtHv3keM8eaffCHQax94LmoPFF2+QuecjNIrShG6zMSnChVOIKfCZZFPvGmWkoXYJ+YYnPka6MorIjC8WGiyM3dHmaCWdnwSxdobyN1cK6F3BrR+00RHZZpKLE24GkGhLujHbEH2jqNZuUkmrCOnnCaDoQ44LmiP5qQK9ge0rxpT4eN4+2y5BPZ07tBEW8Dbtjg5tvhN77jn8p/MMUYja3AYDGhIx/P1mRb5WISmA7e1JXGaUK2KytbdkThudsjmkwPOQ9LGX8AdV1BmOwvheyBjUil3ZGdJ9YG3DSlyhaVysZJNBs0Kvk6CAaack/bSkVPlJAtG3hLfrVjjdpnY5YvpyIQYm7Anaup5YytCGMaO7Eg4sjNzoTUEXqQDUYkmbdSqj3sxCeRPy8AdnLZdjd90EOsDbjp4+i2IozZ8O2K+zQt1UEuDK7cbE0q0qTkvoOm0AajmZ05yZg8nnyQnoE718SAG0CuxjTDQsmhK7zYnHoKDW4jsS2KbBiZYHHkk29qwBfUEbGNis2So/Uj8sqO2j9zEqTCi0zLEJzdp3XItLcHXzNwQyyU/CQUuzSUWBvwrsdZo/VAWz5wGzv+Vnzgwp7AlDMtFJEtFdUPzIc37pCbimdqubQOWaqijRn4iRJG1kOsDXjHDNxQLfK5ZEzfq0IzS3Mbsvl94CaMQL40dibbKlc7mZKfb7ZmJOBUXoZLNOG5ZsmmDHqu/mK+33cNLWDygVRoBm5KS6GVW3fY9FgbcNMbCrb8W1biISMZwMrs3kMhxNoHbnh0ifrAbexpCG/e2+BoJz5wgzCdbV0ywH9nyOfrkzw0Ye9BF08WihQbIjO5s2n0CBZKBzXUBtNFTEUgxNqAm+Zk2kjQC7K+TJANkwr2/ICS7STJ55ULJSsh1YHW+dvaZGgEyS5jI8RrKj24TgjE24C7V+mTmKYhnTFbeiVhaxYiOSv0ghJJZvoxn05Nlgcufz+FwztYWBmeKNnoPUQy4Eqpv1dKrVdKrVNK3a+UshpbMZehijRb6PBdmqV/5VIjGmjKvRaKRR2F7pQdg0YKhXy+xtoobzCr6DBNR83npzbRN3WODUbTKPQAMkKF9HzgBaxaZPqpe5WkQwZBaAOulJoA/B1QrrU+DSgBrjNVMD/IPg5tItYEdF0dmTZUhWaWZhgihXuRqaBfueSYjW0hF/vDW0VkN7fJ9rcRC8XUUj4Xxc98v6dLI5iNnePKzNEoxvpNKr+bpjsm5VHjgfcGBiiljgMDgdroRfKP1mNtgJmO9sDbO/jLO048canl0fLqJh5btYtVO5rFdLyyqZ7n1tYBMvWorD/I79/YJii/hbvfrCalYVvjIWaONRure92u/dz/9o501iWDVag/cIRfLaliU0e2FjNy36xs5Om1u2k92m68zQ8da+O2xZuNZbXKRk3TIX77+lbaUl1Xy6ZQ23yY37xaxe79RwA5Q7qproVFFjIXBUFoA6613qWUuhXYARwGXtBav5D9PaXUDcANAJMnmw2yfqzNmYK3tUdft/z8uY0cPt7O7PFDKBtoNtuJh3v+up2n361leGk/Ljl1lIiOha9uZXl1E2OH9Gf2ePMB559cU8v9b+8Qk//E6lrufWsHIwf1A6B8qtnEGg+tqOG+t3cworQf8yaXMbCvuZwmSzY3cPeb1Qwv7cvMsYMZNbifEbm/f2MbSzY3MLy0L3MNZ7JZXdPMb1/fxrCBfZgrkMHm+fV1/Omv2xlR2pcxQ/px+gTzOl6q2MMfXR2Thw9k2shS4zoAHlu1i+XV+5g7qYzSfiUiOoIidO9VSg0DrgGmAc3Aw0qpz2mt7838ntZ6IbAQoLy8XMRDZGKgtKc0150tn2FmmlCGmUwdZ00ZxoNfWyAiP5XSlPRSvPV/LxOR357S9O6lxLIVtac0I0r7isj3ltdP33QB48vM5Xxs15rZ44fw5I0XGJPpwXND/ubz5ZwzbXjO75gIT/Hq9y5hUD+ZBGBedqXF37mI4aV9RXSAU5cBfUp44n+fL6YjKKJsYn4A2Ka1btBaHwf+Apxnplj+YJJtoS0cJS6UYSYuOuQZNDYy3EvRK52ref63ZJnjz1iyxooqEMe/uxDFgO8AzlVKDVRO77oMqDBTLH/ItZkWNpBSsVNu5gJMFe8BkZkKPu5q2FmVnxOeUWZsfk/rhdfhbxCG6Ud+TgKGkisY/KxjL1c45kyxfh8lAFqQOkTrm37qEV5+GIQ24FrrZcAjwDvAWlfWQkPl8oXs3f4oXdAOD7YnxAGXPkUqf1pP7ni+czXP/5YPKSDJLwc7SRZsjN8T5QCPh0hOKa31j4AfGSpLCP3O1UTnLpRhxuRJz6KDOzLVLNrvi6HQiT2ITsNzkjHn/zyy/CKnAaN0pWLR8MKKLlTmqH2zUFwbE/0+feJVdlIBhceWmboUtjXSJ7hzIdYnMU3OHuKeYaazDskZbOIDzytb6HSn5KrK28RMfOB+9PQsH3i3w2jMDxvR2Gw8JBCOxoZ0tnv5iHVSAaHSp/QMyxXsN/7CUUTwT1s53i6vw9PTHYmLCyHWBtykf82WD9xOBD/Z5aqsP1N+sCc+8DSkfdQ2fOA9JYZRGMTcgDtXYz7wAjN5E7vLfpfvUXbkUyl/HTmsBj+zkCisCL8PiLAaJLPC+MkIE6bcRdskCrNC2EftN3Z8NOaV/zpE0eNn/CYp1QKgY/fZQC3yZZgxCemUUh06YsxCkV9ByFE5pTKWi/rAvUmQWJxxG+PKudrYXzrBJuAxN+Du1evcYft4sVmCybFTrJOZUFVUR4QKFd2Jj1gBXWTAR5cv5wLSxQxJSMU5YkBFFZmWXWAVa6KdNIUfPkZ0+FiJm5jUSPf9MIi1ATcVwN3WJkjP8YHHkyUCdnzgEicx48sDtzOuIPGBxw6mNo1sZpixk28zvvJtZG2R3rAzDclVQ7rMMvs/Ng6/2MrRaiOXbVDE2oCbOoFlk0cq3ZmlZ8jSXHY7J0llN+zi5AMv6vYxIN9Oph9ZHSD7IA2LmBtw5xq1c9vNMCOqQpzPbofmJyZelMsrZQwl2yQdv0XuoWYj272N+NzSZyDCINYGPJfrI8xyz8+DwMTq2G9njkZ18husKYr8IlSqSEtuWaql39laGPF+fOBh5RZMwRdcZFp2x0lMuX0BX7Ij9hm/5Y9E0fUxtmwHs5IJ0GsBG2oP8MDyGiCDhRLQP/FmZSOPvrOLdrcX5+8D0Tr3yxv3sOjdOrY2tDJzXOEMM2Fmh82HjnHrC5s4fCxFQ8vRoqUNU5sjx9v52bMbebu6qbDskE1VsfsAf3hjG8urm0SWqa9squfpNbtZX7ufEYPyx4wOOzt/Zu1unltf58rIIzugzLb2FD97diO79h1i0rDc8cXD8rc31bXw+ze2srWh1ZGTQ0wUbvjz6+t4Yf0e1uxsLhJ7JryOyvqDLHytijU1+4s2btQYN798cQsrt++jT8mJNeeNrQF/Ys0u3t7WxNyJQ0MndPjzsh08v76OMUP6M2XEQOYIZCQBuGtpNcu2NjFqcD/OmzHSuPzl1fu4960djBrcj7KBfZk/fYRxHRW7D3D3m9WMKO3L5bPHGJf/5JpaHl65kwllA7hIIFvR3UurebOqkdGD+3P+SebvwcLXtrK14SAXnzrKmItmW2Mrv3tjG8NL+3Ku4Xv69Lu1PLTCae+5k8qMZQ/y8LvXt7Jm535GDeoncj/BeWh6dZDKcAXQcrSN21/awpD+vfnYvAliesIgtgZca+jfpxdPRMhSktKa6aNKeeHvLzJYsq7QGk6fOJRHvyGT78JzJd31xbM5TSBllaPDuf7np+Zy8amjBeRr+vbuxdLvX2pctid/9vihPC6UTUVrzYIZI7j7S+cYk+m1+Y+vOY2PzhlnTK4j28msJNfecPbUYfz5q+eKyHd0OA30+vcuoZfgxol23Uzf+sApfOWCaWJ6wuDEWg8EgBMVLzp90MamhHwMbdmNKBs6bGxexi2KoiRHOwhzI+ypVBuUWbDD/wY7TJegiK0BN7EjbIvXaePwC8gdh+6kI6ZshTgmuvAT5zospCPr2Yjc59H65Pnf8hOksIitAc93ICNIMJkgs7JowXYCbmAFVBaOThlMSRDOfbgZW7DyB9URmP4Y6h6YZRj5pSVKpGmLiqDywwSBCkNpDdM30yE7ZORHQWwNeC6jGPQBaYuYL82dNhVSoLAO94XgiUDpE5iSCiRm4J5dlphhFou+GV2+fPYxG8HhPD1A9wQ7KYLYGvCU1pE3Lvwuq6PeN6es/r4bRlXQpXaY+vj1gYc1CtrnEf3wAcv8zaDC3mo/D+mgZffje5Vsj0jUO5+Ha6IMrSCrtih6/K6EuuOYfWwNuIlZrRPlzc4JLhv58qQz8Tg6BH3gkkwC4dN6EpmEJNtcPHVdysbJZnsraEh84EZhYslqK4aCdBYbG5sscY9aF0f5kkHWgvTJcKdHbeR/tWNUExaKAFI+N40KwVaAdhuzHZCPQiipQxPfKIRS8qWj7EnPkOWje9qZgaf3mE48Cx5bA55vlzvIbCEItSxKDIXgO/LBEGYGHrQ6QfzsYdoqMKUzcPlPjDjjQdrGb5TME3GGHJS2GTZGTGAWSoRYSb5CulimocTYgBvwgVtagolzbt2raFWs8MBFRHsKRGltEowmSd/riZK6LgoSH3iMDXiuGUSYXUETFwoAACAASURBVH5fzIdgYnPq8Ws8JBkiUXT49gOGbCy/yS7CPgj9GpSwY9SP/KCi/fjAw/ZNPyuSaAwRn2MrCtMlyEMigqKOvl/EWnaHeY+xATeRiccOtVN6+W7jSHGHDqns5cKzKRs8c9Pll/S9+qVthldg54Rk4gOPKUwMeBvB5j090sYDLB0WijELRfzouOG77Jd/HE62/1VhuD0NeRaKvVhGzvUE9KBEM+BKqTKl1CNKqY1KqQql1AJTBSuGuPnApbPYgHSuSucqmVxXtvwn1tFxfzKdqwQ/vkf4wC3RgL0NoBPRBx41nOztwHNa608opfoCAw2UyRfM8MDtLcEkA01ZjUYoVQ8LD7m4GSxJ/rH4pMLCBqO9YHTOtUcZcKXUUOBC4IsAWutjwDEzxSqMtvYUT6yuZeqIrs8LP4u9o23t/OtTG9ja2MqciWXmC+jivmU7eLOqkd37j3DKmMKZeDLhd8m6rbGV/3ppC1vqW4Bgmyh+NFTsPsCdr1bRntLs3n/E1eGDKhegHIeOtfHjpzewbFsTvUsC0M58akmlND9etIHa5sNMH1XqX77PSlQ3tnL7S1vYd+iYMYP12uYGHlpRQ33LUfedwoKDtPcvX9xMZf1BVtc0hy5fIfzxzWqWVzfR2HIskEspSB1qmw9z6wubWFG9L7DTKgi7yLMTNfsOA35XuHZ5hFFm4NOABuAupdRcYCXwLa11a+aXlFI3ADcATJ48OYK6NLwGHdi3c/H9dpjK+oP8edkOxgzpx4UnF8/OEnZgLnytisaDxxg9pJ/vLDBBVL2ysZ7HVu1i6oiBnH/SCIYO6ONThz8tz66r44nVtR2G78zJZUzIk9orLTsYNtQe4P63axg3tD+Xziye6Seo/N0HjnDX0mpGDe7HBX7udUD5SzY592DayFIWzCicNcfvbPHBFTW8sL6OScMHMnfi0JwTlaAywXmY/fLFLQwb2IdhpX15f7E+GaLf/3pJFa1H2xhX1p/zirSHoyK4kqWVjfzlnV1MHj6QD8zylx0qzBDesidtJ2aNG8LMscXSIYZQEhFRDHhv4EzgJq31MqXU7cD3gR9mfklrvRBYCFBeXm7k8eQtLb920fRQv/dmVz/52Ol80GcHCIOUhg/OGsNtnz5DSL5TkaduuoDB/f0Z7yDw/Lovf/di47I9eMvTWz85VyTVWcpV8L0PnconyyeZl++W//Fvns/QgWbugdaaqSNKWfydi4zI8+D1ly+fP42bLjvZqOxMHVfOHce//80cEfmQHr/333AuE8oKTyhM6Pnpx073/aCwjSgezZ3ATq31Mvf/R3AMujii+nxtxTaQpq5JhhsF+UQUng6Q49DaaCNHgUGZKan4J841qOwgsy4bfmkb4ZM76TnxXN8dCG3AtdZ1QI1S6lT3rcuADUZKVQRRaT22NiWkT2DaCTAlIzutQzbeh0a2jSSoflIMEem2BgsnarE3fk/kTDweorJQbgL+7DJQtgJfil6k4ohKadMCs6Z8euId4tXCLr8g1xkspIIToJhpZNtdln0if+BF+qGc1uPgBLbf0Qy41no1UG6oLL5RaObph8ERZlCHDbYT6mi8b/nhtxT8tFPoA0gBipVeTQXT5LfqYZfB/u8BgeT7KXfQdpduiyAIS+8NFoTOfRGmHgH0hIkGmQSz8oG04cpq2IAny+J+UszG4Rp7y1QZ+bZ84CbbKaVl+PZ+oxtG0pGS3zexce4B7LRXVMTSgEf1OwadgYddEobZ0AlKC4Pg7eBXRRgfeND6SqeDC/qwDrsSMBkULciDM0hpgz5swvR7HUA+hAyslgpuwG3o6Q5XS8wNeDQfuHx7x98HLh8S1EFsfeACs0GpvYewq5EgbgHxIFnI95lsPSeyDzyWBtxveMf8v3eu8nQneX8jSAaY0lY2ekGehSJVjXSURpMyZeinNiYudoJYOVfpzdITOQqhh1gb8PAZ0HuGD1z6QWQjWqOXDk5sBt6Rbi4+PnCpIE1WfOAW+kzHg0jYeklGgzSFWBrwQksbP6u99O8D7C6HiHEQ1gXhe8kaYbbjt51CMQpCfDdwaiyf37O3Serz+z5KrgmYjsxna6QDkglvrIfqM/57TRQXaqC+6ekJ0F6WSSgxNeBxOokpOtuRjwFui4UipSbsYPcbUMxLpWZyhu83O1FguRZchzb2TWyOXxt6oiCWBjwf99b/Lr/3e9ndZceFHJCFEuC7YR8QQdopqPygxdEB3WFB27PDB+6bhRJIvNNGPr/rn/0TRKjP7xHcBx6KuRFwVRhmaIXxgYdxt+alKxvUERWxNOBRWSi2YhzIZ/6WPxZtY6MX5OKMS7NQJFYp8j7wcA9Bf9+1uLkovsHuXJMZuGFEvoHCg7pDjRb2N0rTFIUMSbYOkDWwIOiiwXzZg/rAg8gFG5uYcvIzYSPrvQ09URBrA/5e94FL+xt70knMOKWCk9rbsMGJtxE/J+wBtuB6cPWcuAY8ajAr62hPaX70xHognP/slU31/M8rle7vZW7M4g17eHhFDcfaUyI6Dh9r5wePr2PF9iaxxeojK3eytHKvmPwjx9v54ePr2LSnxX3HrKaqhoP84oXNNLhZbUyPwV8tqWT1jmY272kxJnt97X7+5+VKKusPMmfiUDNCgXW79nPHK5UcPNrmvCFwU7XW/PjpCke8UKe545VK1tQ0U9Vw0NUjo8hrr44sVCeu/Y7fDLzuwBG21Ds3cMboQV2/UMRd99g7u1i/6wDnTBvO5AKZTrqIDcAPenhFDUs2N/C+sUOKZmnJqatIJbbUt/DoOztpT2munDM+sHwoXp9739pO86FjfOT0cSFkF2+sbY2tPLxyJ3sPHuO8GSMCB+YvpuK1zQ0sWrubfYeOcdaUYYFS2kFxOthvX9vK8uom+vcp4aoA96BQuV+qqOfZdXWMG9qfy2aONiIT4IUNe3h2XR0NLUeZO3EocwXSCB461s4flm4D4JxpwwP/3s/4+s2rVSyvbqJPSS8+OmecWNAsr72OHG8PYSfsEgljNwP3lk//7xNzGDmoX6fP/DyRU1ozafgAHvraAt86wzATTho1iGe+9f5gPwRfsyNvKfzja07jkgADPa2juBKtNWdPG84Pr5xlWjSQdp3881Wz+NDssQLynesj3zjPd6o5CMbQuXbeBG65enYA2YWle23y7Lfe758h5eM7Xljj5759oS+ZQWR78Mr+g4++j/NmBMisFIhFA9fOm8g/XxWsT0KwMRy6vbphph67GXj0OCjyPjot7PuWyALTVYfskWvpKqSP6MvIl/J9g/n+aSezknOVTmBiw0jaaC9TiJ0Bj8oqsBOrId5hZMFGGADZHX5p+RIbyFIP/siZlXx4BezEWbFD6bOlxwRiZ8DDHr3u+L2Fk2Jhj6D7lm+BRSO9UkkfUxaWL8ZuMT8RkOqbdladzlU2+qY8KwrstJcpxM6Am5iB24lCKDl7da7yM3Ax8eL5GeUP8JiXLdXmdvJUysdZsZEwGey0lynEzoAXCz9abLUX1rcbLAhOtKVwsY1sE1m5i9VHhzhG71c2RA/DWoypEyXdnB+EDbVbqFipEKEX/MAJT3ti+8D93C0TLiY/p0qjtFcSzKoIoi7VtNZiy3YPtnzgoi4O4eWqdFqsqPKL2f/3lA/cB2y49RIfeFfEzoAXWhr7olTl+W0hBA6gFKED+NEVdbD4+VlYF4HfX4Tdy/AtP+yD3md5QoUxKPL9UH3TFyU05L0MFNLW/U3g4G0BdESYGAX5Vej2CvyL6IihAY9mvKSynXTVIe8Dl6ZsidIUU9I0P1z5cj5w0/fYSQhsVKQjN+K99OcSszMDt2EkbWSiMoXYGvCwLWxjI0Q6lZqNoETycVacqyRPGyRzbZrf6JKKgWIls5KFOCtgcxMzHhY8dgbciA/cgj8wzqnUQL6dbPnA5dLNSR24MSrSlWsj8JOdFZUNw5r4wAVR7BBLsVgEYWeWQWIcOGFkA6vo9PtCMDG7LFafKLNBP01lgs9fTL4kBxzCpvQqXHCJvmmLPw0RDGzRcWtmReWvb4ZvL8uhUOJnwKNnpJc/jitO27J1bFlMunyc7qh8/0KGVsr9IzkDt9Hnwcaehoz8bF0x8aDE14DnMpB+Gj3UwA749SgdwG8dIAILxcfvwroI/P4mdIZ0n98Puwz2x9AJ1/7Fvh6GfurvXoZ7mAX5SVgXR9DgZ2EfykGDWYXS0w1WP7IBV0qVKKVWKaWeNlGgYujYwgzNQrHDibXhA5fmacf7JKbciVupskv1TSdJcvjf+3ELnKjJqcMganvZhIkZ+LeACgNyfCHy5pcwu8JVYWUDUDbioXA7SQ9IwQe1lDGRisFhwwcubWBtMK8ydcWFhRIpHrhSaiLwUeCnwHeMlKgAUinNPz22ztUd7Lcrqpu445VKNtYd4HSD2U4y8ae/VvNSRT1bG1q7xCo3gW2Nrfx0UQW79x8G5DIK/emv1dQdOCLygNiyp4WfP7eRugNuthPzKnhoeQ2L1u423j6V9S387NmNHG1zcm2ZbJ+7l25jyaZ6oyV+eeMe/vjmdjbWHaBE0CCt2rGPnz27EZCZVDy0ooYnV9eKyfdQ3djKTxZVsHZXs2h7mUTUGfgvge8BqXxfUErdoJRaoZRa0dDQEElZY+tRNtY5KbhOHRssw8riDXtYsrmBaaMG8eHTgmeZ8YP7365h1Y59zBg9iA/NHmNc/ltb9/JixR60hgtPGcWUAJlCguC+ZTsAwiWLKILXtzTyYkU9AO8/eSTTRpUa1/HgihqaDx3n42dNMCp3aeVeXqyop6nVyfIzP0TmmXy47+0dHDrWzrXzzJX5qTW7+WvVXsYOHcA1BuVm4+WN9Szb1sT8acM5fYL5ydHDK2pYXdPstnnwDFd+4Y2vEaX9RNvLJELPwJVSVwL1WuuVSqmL831Pa70QWAhQXl4eiWTjLdP+7drTGT24f+7v5PltSmsG9Cnhif99fjjdvsqnOW/GSO78/FmhdBSD52e8+0tnM3pI7vqbgNbw4dljA2XK6fT7ggwO57P7/te5DOnvP1NOVx35kdKaeZPL+MnHTg8tP5cCr+x//up8ygb2DSc2T8FTGi44aSQ3X/G+4DLzvJ/SmvFl/UP3eb9IaU3vXooHA2S5ykbh+wlnTCrj3q/ODy3frx6INr4sswgjzcDPB65WSlUDDwCXKqXuNVKqPCi2+19o0WPjGK40RdHrYNIVSQkG/OoIxCUjHpA7bZuOoCgh23wHtRV+VZp2ZzMTDxCbY/QQwYBrrW/WWk/UWk8FrgNe1lp/zljJciAKFzRKhEC/vzKx8Vfo16ZOL/qhs4U1Un5kQ/g6+A32JUGx7Ng8DjlqCss2H0ApanRD76fFDh9F2Xz1R9uM/iAKFiRO1k6YRKx44FGOR0eJb+1fh+xMwRaVSiNIB3Ov0uF2pTLbgGS8EtMy7dDuxMNTWAh/AXb6pmkYyUqvtV4CLDEhq7Ae5xr2uLFkthBHh4UIgdjgscvnqpRecovwqQXbP04ZfnLrkT33YIf/7R0SjA9iNQOPMoBs+cClw2mCHH0wrUeuHjqCG8wvnHaKnw/ctFQbmXgcPdJ7GvIhoB09zjVOM/CYGXDnWsyXmPu3EWcJvk6jmbn5+YITRfXBdpaV/7PIcUQKyU6Z8eMXUiK1pDexeshX6iirt3xNYcNt6OmJej8L90dzK9tCgb9SBsZXkKB3JhArAx4tCpz8jrxGdqZgzQcuOPDj4gMvNAzj4gO35noQ3/uxm2Q4PvPvmBnwYnGwCxnoaMwEfz9MpQzslhf4uSkfbLEyRjKARX5mIxBXFBpkIZdDevUgIDssC6VYn48wwj3RRUP3RthfkkwJ11lR8a9EZkh1g+WPlQHvcCGE+K1jXM2WJxvyAaCcqw0fuJQGW+ngZPzUzlWGY25+FquR7ytpPXKwxwN3rokPXAhRlt/SO+WODnl2BdiJ7SyZKUf+QSrTRpIBlSRcfD2HhWInuJSt8WUSsTLgUZbfToRAeR+4jY5mwwcueRLTTjTIeM3ApXzg9k5iCu5pYG8zFku6TCFeBtwNmVXQ75dn+ynqMszP3rKpjpw3tkVEH2xnHYV246PVoxjDxUz55XV00WlCboFYKKHTeOURamvzz3koR5VRiFVk7oFcqG9K52mVQLwMeIQlTk84iWnDfwxuPYRkO5xhCz58AVqbdJII4z5wSzNw+X5v1wceH/MdMwPuId8TslDD24uFEkpFhq5CTAVDLJQin0drq2IMFzPxOQohyoZ18Xgl4eT6kx2ChVLgs+grEefHxVaf0mPLhA/crx5IYqGIIZIP3IbvVXizxRtI8nx2wWw22FmiSvnAkxl4bj3SvP7EB54bRmKh2MJdS6uBYA28se4AP11UwYbaAwwdED7+dCEsr27itsWbOXCkzfhTWGvNDx5fR1XDQWqaDot2Lq8eLUfajA/8DbUH+PdnK6isPyhWh0dW7uSRlTU0tBw1fh9e29zAna9WiTzYbnlyvdPmhkpdWd/Cvzy1gXW1+5k1bogRmdnQWvPDJ9ZRWX+QyvpW+vU2PxesaTrEDx5fR33LUTG3277WY/yfR9Zw8GgbNU1upivDHfTZtbv541+r+c9PncGEsgFGZcfKgK+v3Q/A7PH+s368VbWX17c0cubkMi57n/ksOQBLNtXzZtVe5k8bblxHSsOfl+1gQtkAJgwbwKUCWXI8vLIxXQ/Tet6sauT1LY2cNWUYV5wukxHpyTW1rN25n3mTy7g8ZDKKfHixYg8A1y+YalQuwN1vVgMYa/Nl25p4fUsjZ0wq4+ozxhuRmQ2t4d63nH45fVQp580wnylnVU0zr25uYO7EoSIZrgAq6g7wYkU9M8cOZsKwAVwyc5RxHc+sq+OtrU0ix+xjZcC1hitOH8uYANkyvI2Ju754DkMHyszAUxr6lvSKlJEkv2ynAp85ZxI3XnqycfmddUHf3jL18PruH798DoP6yXQ7rTWnjh0sdh9GlPbllqtnG5etFNx0yUlccPJII/K8Pv/b68sZNdh8blZHh6PkurMncdNlMv3SM3i3ffoMpo8aJKTDuf7rNadxjsEUeZlIac2MUaVMHGY+BWLsfOB+joFn/waIvMNQNAiOwVVXpqo088bssi5XdWTZJ+FP0eaCNEskm5pnjiLaWa7WOrKvOrst0nHzQ4ssiijJVXIh1+iS6vu5dZiRl29cSe0RxMqAF7WTOT5MB4CSKFFaR5zDr3qQPPFm45hyKiV5H2SobBL311jERxe5Ji9WjKt77sNGeApRHYJhPOJlwAneKdPHn8O3YLGfmnzCZouRiECYT5SJB1Fe2ZiZ6RQ7xBU99VYe2UJtEzU6Y2GZoUQWlJ2NyPS+QtRKQzocPbllRMny5VuH4AntWBnwMLxWKzM/QRqVrSw8ji65etgIhWuCh59ftswglLi/tgKGgfTsWN4VZKtfSt2LGBrwYA1hK4WXvO9Y3oJLZEbvkJ2yMRjlMtBIPdwkXBFx9IHngsnZcT6Y3pvJBcmQBvEy4CkCt7QNH7LkQQMbA8WD5GaLtZWQUI9O5fADm4CID9xCTA87OnB1iKmw1i8TH7iLYg2db0deMkOL6YTJnZgKQh0s5245hgJl5RBuOhRrTh2SJ2ENPRy69k/nGqXc2S1ha+MeTKY6y/+eyXsqzdjJFVgsYaG4SHzgsrDBQpFdDssZlDj6wGWPt/cMH7i9GXhiwItmWsn1mQkfV7HfmvSBZ99nCR9pvjY0sdTLz+IwQ8MrJMIEjz1fGU1kXi/cP8NKzy8zKjzJuTna+bQH1VGAVWRw/6dQvwQzD4l8IiT3yGJlwHWIZawNH7JskCPnaieus2y4VBsZkeLHQnGuJkXbiGvdsXIQ7Jg9xQcueU4kVgY8lKF0Z36yUdnkdpnT/Gl5Cy59IEn6ISTJt9UITQIE/bxx8oHn1iH/IJJMk5epI/GBE46OI+mf7tCRijd/2oO0D1z6IeSceJO6D4kPPBN2fOC4OuRn+dInShMDTnEfeP7fyEIyY4jNRKsm/Lz5IBlnxYPofRA6Di2xx2Fz80/yfIKpOEaFYIczLzewQhtwpdQkpdQrSqkNSqn1SqlvmSxYLoShuZmagRfaFzI9y+8czMq5mp7p5KI7mQoGlXvTy+wMNpcOk26abPmmyt9VrnONUu781DgbPHAz8vLdT5M6HD2dNZnms+ejQ0qtVKLE9WwDvqu1fkcpNRhYqZRarLXeYKhsXZBKFTYwueNCRJ+VFY2AaEBHhrbOsiUGY94t+ehcZ8k4K4XkgxlfY74ZpeMDjyg7T/8kgux8sVDMtLVKC8yhAwwYviL304gOCvdLR0dkFQXjAPVSMs6O0AZca70b2O2+blFKVQATABED/vz6Oifjjc+GXrdrP//69AaqG1tFlkd3vlrFC+vrAKjee4jSfiXGdVTsPsDNf1kLyB71fXD5Dh5cXkP13kPGY3U/snIn9y3bTs2+wyIzws17Wvjh4+s43p5i+95DnDTaXNzoxoNH+fsHV9N6tI3K+oNG4zkfOtbGjfetoqHlKGBmCf/kmlruXrqN2uYjorPvB5fv4J63tjv/CKl5bt1u/vimo0OiKu0pzd89sIp1u5wkMRKuoH9/toLl25rYVNfCGZPLjMsHQz5wpdRUYB6wLMdnNyilViilVjQ0NITW8eIGJyPK5bP8ZVpZtq2Jt7c1MX1UKV88b1povfnwxOpa13D3Zvb4IXz2nCnGdSyvbmJ1TTPvP3mkWLB5gEVr69iy56BTj/mTjcp+du1uNtW1MHPsYL5wnkwbLdvWRO+SXpwzbThXzzWXgWZTXQuvb2mkLaWZO6mMT589yZjs7XsP8fLGeo63p7h05mjOnR49o80L6+uo2N3CyWMG8eXzp0YvZB48s7aO7Y2HuGzmaOYL9cvFG+ppOHiUv50/mQF9zE+O9h06xqJ3d1PSS/GxM8Yzrsx/khi/eHTlTmqbj3DmlGH8zbyJxuWDgYw8SqlBwKPAt7XWB7I/11ovBBYClJeXhz5hoIEJZQNY4DN1k+d6+O315Qzubz4Tj9aa8inDWHh9uXHZaR3O9fbr5jG8tK+gHs1JYwZxz1fmm5cNTB8lIxvSbfQ/n53H6MFmB6En+4dXzuLsqWYNlSf72x84hQ+fZib9mwYmDBsg1taZek4aM4jff/FsQR2aMUP68dNrT5eR77b/l86fxufPNT+x8HR8YNZofvIxmTpAxBm4UqoPjvH+s9b6L2aKlBtBGQbSPFXJAFYebERKA4onyogASWYImD2tlw3J9pdgithg+tjS4/RJeRqkZD3CsOaCIgoLRQG/Byq01r8wV6Tc0D6ZHt6NMb9L3nX3WvLQiKPDuUro6cx0McwQyWLRSDxEPR2Sp/UkQ72a5fe7fV6Qb5wJib6fHZwsTNwjf3o8+c5VMkic5MlgD1Fm4OcDnwcuVUqtdv+uMFSuLgg6k5M+BKAFZXfo8AyIMFtf9GEkNBA9SIU19XJVOrKNigakDvAIpX3LnrxYeFBIH8CzlZBC+vBaFBbKG8iv7jP0Fb+hmZ+aSuGVLdeD1GDJlCl1CjNbmkl3UPaS0e/KKYp8MKgjs/0N0tiyYYqG17nPm27r3NCY87nlp/eZHV/ZsozRIDN1ZDeKhVPgsTmJGdoHLvSM0TaOhtv0gQspkT4JK3laz0uqK3kC02S5pRIvZ8OGa0B6fKVMhVMspMPC/YiNAQ86k0tn5pYqj6xrAOzEgnD0SMZAkY1CaMfNIeHDNy/bRtwfkE1Q4EHKB+7BRoyhE90HbhVBb6jEEqlzeSz4wA26gQrrkQ3GJcpCseDmEPErCzx4pI2eBxsMLNOuty7yYx6F0EOsDLgfd0iawWHW+GXHOJBcHnm6JGcJmdUxXZfMTS+pgWiHqWP+4SBR7kxmhY2ww3Ziu8tsAEr3m8y+b+N+xMaAB33qS4eJlOapAlYyuYN0QgpZP6BU5D2tZZOBSOxvSPnAu05ebOz/yHO0QXh1mPjA0/DjsujM4DC4nMwhR8oHnilSapbQZUfeYF2yZRufrXUpe269JsSb9oFnGj1THPNs1pJRVkUhhogpHXnedxKFG1JCLvaSeY5/dmUSH3gGgt5QaR+aTR+4jR3/2GWy8eSLbjTiyjYuumMtH0sfODayK8mnOQPprEWJD7wDQY+lirMfMDtDyAUb2UIcPXIDX/5AhnOVzJYj0f4d5TbY8DYOjnh64uoDT8t3rnHNRu8hNgY86FM/pRHmeCKrALu8Xqm6SNdBkisvSDEX8oHbOVknmbquQ4ctH7iQfBuxViBGBjzo00yap22DBy7tBkrrkZyBy87WbGw0xmV2b6u/2DigYisEg9zmvXNNXCgu/N7QDgoecmnOvP/lbo6jTNSnmVEf03XRXWQbE50hN821VEo2YJZE26TPKZiTmTK88ZctPxPGU/xZGl9dg4mZlt9Zj/Qkz2z6FSGsr93P61saOWvKsILfUyjqDhzh2l8tZcfeQwZ3yh1JB4+2ccOfVrD/8HGaDx8XjYVy5Hg7v1pSZV4BXn00f/prNQ8ur2Hb3lamjyo1I9stf/OhY3z93pVU1h9k3NABRmRnyr9v2Q4eW7WLPQeOGl2meg+CK//7jY7sRKYZLvct28Gdr1Z10hdepvP7J9fUsmL7Pi44aWQkeblkA9z+4hZe2OBkoNra2MrUEab6S1rHIyt3ctfSbQBUN7Yya/wQIzocPenXtz6/iWfW7e6iP7IO9/oPD69hQ+2BLnolEAsD/s6OZgA+OGuMr++v2tHMBSeNpHxqYYMfFNv3tvJm1V7mTBzKJaeO5qOnjzMqPxO1zYcBmDl2sJiOxRv2UNN0iAtPHsnHzzSbMaSqoZW3tjZxxqQyPnGW+WwkWxtbGdy/N/OnDedUgTbaf/g4Z08dxvtPHsm4oWYTRbxYsYd9rce4dt4EThljJgXcko31AHzKYNagTDy7bjd7W48xd+JQxg3tz8cF7ulLbniNJAAACnBJREFUFXuobmxlwYwRjBvanyuExteitbs5dKyNK+eMY55AqrPHV+1i4rABXD5rDJfO9GezwiIWBtxblwQxBLd+ci5jDQ88b3l04yUncflsM1lU8upyr9+4eIacDg2njBnM774gkVnFqcF3PngKF54ySkA+zBg1SKjsDqRka62ZNqqU2z59hjmZwOThA42mlOskX0P5lGH8+nNnicj3dEwcNlD0njp6NOdOH8Ht182TkQ9cOWc8//ChU0XkZyIWPvAwGwKy8SvsbCxK65INYuVcJctvg/MsAQl6mTQH3B51UFSFq8fGuBIT3wkxMeDBKTmix5+tUftkdaUEeWc2wgDY4DxLQCLErjTnWLKvpHXYi+UiyQ3Q0qfXMhATA+5cgzw1T+Qg/EX1aFkKGzgHkaSO+mrsPIDiOAP3BrjJsjv3UjC4GnaiA1qh5iIca9zCCc9MxMKAh0ktJnr4wsZxZRsdQfQIvbwLKK4zcJEwo+IGNv7H5zv0CJ97ADv1gNgYcOcaJJiVRFAfkQA42brcR4/kQQNPpMyszREodYpR5XktId+4bK/dUwapiRn30rRRyiyjlNshU6ZNH7gUBdhWFi0PsTDgYRoldsGN8uiSVCWdiQfMxvrIhq1ZjmlIxPkIGisojHwbiYzjHstFIsZNIcTCgIfyPYtsYoYoR2hdNlgoNo4Si4h3ZMei93aFxOlUqVllh3wL8U9s+MBBtt9ri25WiIkBD+N7lsyRaNUHLniHJP2aNtxNcZ2BS/jAbcRB6Smxf0DuQWGL6OAhFgY8DP/a/ADR4pnuO3QhH2xHa7OB+bvKdl7H08QKQmvjs2XvXko+7KX90x7zSrq/OHrkJmHtHn1WRnwXxMKAh+EUy+ZINC66C1IWDKAVH3hMZ+DS/HvTZZf0gWutrR3kseWejPMBtkzEwoD7XZZkfmyy/TxRNjYo0uWWM4BpVo35GX6mbJBIB5eWJ5LSTnmyBdrdvZps9w6ZSGeXkWVEpXUYV+HqSQuWctUoVOIDz4VUiJmvyFF6SynOwF7GEGk+rOxBHsnZvZhokc06+ZOYPccHLnroKZmBd0WY1GIyQfi9VxY6mYWj6JLZcmycxBSVLUrJM9+DpDMf2cgOZYsH7kQFkHUdxmIGrpT6sFJqk1KqUin1fVOF6oIQN1ai/cKsBMLrcq5x88V6sBGMS/RAVczaXT7zUU/zgcvJhhjMwJVSJcAdwEeAWcBnlFKzTBUsE2Gih8luYsrfHBtH0SWXrDboVKIcc+F2N/3wkeCWd5KPhRhA2KPoSu1jpfu9iPguiDIDPweo1Fpv1VofAx4ArjFTrM64563tHfQcvzDdEfa0HOUHj60TkZ2N5kPH+D8Pv+voEtKxdtd+tu89JKKgqfUo//rUBiC+bo4+JTKyK+paqGw4aLRdnltfx6odzWLt8aMn13Pg8HER2R5++/pWNtQeED8s9JU/Lufw8XaxnlPTdMh5YWkGHiWhwwSgJuP/ncD87C8ppW4AbgCYPHlyKEU3XDjdlwH/2BkTOHC4jRmjBxntCH9z5kQOHWtHo7mwfx9OGSOXJeeaM8az9+AxNJrzTxrB7AlDjev43PwpDB3QB4BPlZvN4HLtvAnsP3QcjaZsYF9jqbc8lPYt4WsXTqdm3yE+abjsAAtmjODaeRNYMGOEcdmfOWcyA/qWAMGSkxTCDRdO5+3qJgA+cprZDDYnjR7EZ86ZxP7Dx1EorhJIFjGgTwlfv2gGO5paAbh67gTjOgDmTxvO38ybwJG2dk6bMJSPzjGf7efzC6YwrLQPfUp6cdHJMklMsqE6ksMG/aFSnwA+rLX+qvv/54H5Wusb8/2mvLxcr1ixIpS+BAkSJHivQim1Umtdnv1+FBfKLiBzCjTRfS9BggQJElhAFAO+HDhZKTVNKdUXuA540kyxEiRIkCBBMYT2gWut25RSNwLPAyXAH7TW642VLEGCBAkSFESkrPRa62eAZwyVJUGCBAkSBEAsTmImSJAgQYKuSAx4ggQJEsQUiQFPkCBBgpgiMeAJEiRIEFOEPsgTSplSDcD2kD8fCTQaLE4ckNT5vYGkzj0fUes7RWvd5XinVQMeBUqpFblOIvVkJHV+byCpc8+HVH0TF0qCBAkSxBSJAU+QIEGCmCJOBnxhdxegG5DU+b2BpM49HyL1jY0PPEGCBAkSdEacZuAJEiRIkCADiQFPkCBBgpjihDfg1hInW4BS6g9KqXql1LqM94YrpRYrpba412Hu+0op9V9uvd9VSp2Z8ZsvuN/fopT6QnfUxS+UUpOUUq8opTYopdYrpb7lvt9j662U6q+Uelsptcat87+4709TSi1z6/agG4YZpVQ/9/9K9/OpGbJudt/fpJT6UPfUyD+UUiVKqVVKqafd/3t0nZVS1UqptUqp1UqpFe579vq21vqE/cMJU1sFTAf6AmuAWd1drgj1uRA4E1iX8d5/AN93X38f+Ln7+grgWZyslecCy9z3hwNb3esw9/Ww7q5bgTqPA850Xw8GNuMkwe6x9XbLPsh93QdY5tblIeA69/07gW+4r78J3Om+vg540H09y+3z/YBp7lgo6e76Fan7d4D7gKfd/3t0nYFqYGTWe9b6drc3QJHGWQA8n/H/zcDN3V2uiHWammXANwHj3NfjgE3u698An8n+HvAZ4DcZ73f63on+BzwBfPC9Um9gIPAOTr7YRqC3+35H38aJqb/Afd3b/Z7K7u+Z3zsR/3Cycr0EXAo87dahp9c5lwG31rdPdBdKrsTJMllPuw9jtNa73dd1wBj3db66x7ZN3GXyPJwZaY+ut+tKWA3UA4txZpLNWus29yuZ5e+om/v5fmAEMasz8Evge0DK/X8EPb/OGnhBKbXSTeAOFvt2pIQOCcxCa62VUj2S16mUGgQ8Cnxba31AKdXxWU+st9a6HThDKVUGPAbM7OYiiUIpdSVQr7VeqZS6uLvLYxEXaK13KaVGA4uVUhszP5Tu2yf6DPy9kDh5j1JqHIB7rXffz1f32LWJUqoPjvH+s9b6L+7bPb7eAFrrZuAVHPdBmVLKmzRllr+jbu7nQ4G9xKvO5wNXK6WqgQdw3Ci307PrjNZ6l3utx3lQn4PFvn2iG/D3QuLkJwFv1/kLOD5i7/3r3Z3rc4H97rLseeBypdQwd3f7cve9ExLKmWr/HqjQWv8i46MeW2+l1Ch35o1SagCOz78Cx5B/wv1adp29tvgE8LJ2nKFPAte5jI1pwMnA23ZqEQxa65u11hO11lNxxunLWuu/pQfXWSlVqpQa7L3G6ZPrsNm3u3sTwMcmwRU4zIUq4J+6uzwR63I/sBs4juPn+gqO3+8lYAvwIjDc/a4C7nDrvRYoz5DzZaDS/ftSd9erSJ0vwPETvgusdv+u6Mn1BuYAq9w6rwP+2X1/Oo4xqgQeBvq57/d3/690P5+eIeuf3LbYBHyku+vms/4Xk2ah9Ng6u3Vb4/6t9+yTzb6dHKVPkCBBgpjiRHehJEiQIEGCPEgMeIIECRLEFIkBT5AgQYKYIjHgCRIkSBBTJAY8QYIECWKKxIAnSJAgQUyRGPAECRIkiCn+P6NCVtshlZtxAAAAAElFTkSuQmCC)
"""

# This code cell can be used to debug your code.

# YOUR CODE HERE
simulation = Simulation(num_stops=5, num_buses=2, person_rate=2, trace=False)
simulation.start()
for i in range(100):
    simulation.step()
    print(f"\nState after step {i}")
    simulation.status()

sim = Simulation(trace=False, person_rate=2,
                 num_stops=12,
                 bus_nextstop_time=2, bus_max_capacity=45, num_buses=1)

sim.start()

for _ in range(5000):
    sim.step()

sim.plot()

"""Here is some code that will help us test your implementation."""

def check_all_stops(locations, num_stops):
    """The bus stops at all stops"""
    for i in range(len(locations) - 1):
        if locations[i] is not None:
            assert (locations[i] == locations[i + 1] or
                    (locations[i] + 1) % num_stops == locations[i + 1]), (locations[i], locations[i + 1])

def check_occupancies(sim):
    for bus_idx in range(sim.num_buses):
        for oc in sim.occupancies[bus_idx]:
            assert 0 <= oc <= sim.bus_max_capacity

def check_no_ghosts(locations):
    """The location is always defined, except possibly at the beginning."""
    for i in range(len(locations) - 1):
        if locations[i] is not None:
            assert locations[i + 1] is not None

def count_tours(locations):
    n = 0
    for i in range(len(locations) - 1):
        if locations[i] == 0 and locations[i + 1] == 1:
            n += 1
    return n

def check_simulation(sim):
    sim.start()
    for _ in range(10000):
        sim.step()
    simulation_time = sim.simulator.time
    min_tours = sim.simulator.time / (4 * sim.num_stops *
                                    (sim.bus_nextstop_time + 10 * sim.bus_geton_time))

    # Checks the bus tours.
    for bus_idx in range(sim.num_buses):
        check_all_stops(sim.positions[bus_idx], sim.num_stops)
        check_no_ghosts(sim.positions[bus_idx])
        assert count_tours(sim.positions[bus_idx]) > min_tours
    # Checks occupancies.
    check_occupancies(sim)

"""Let us test your implementation with this one-bus simulation."""

### 20 points: Tests for one bus

sim = Simulation(trace=False, person_rate=2,
            num_stops=12,
            bus_nextstop_time=2, bus_max_capacity=50, num_buses=1)

check_simulation(sim)

# Let us check that the times are reasonable.
travel_times = [p.travel_time for p in sim.have_arrived]
total_times = [p.elapsed_time for p in sim.have_arrived]
wait_times = [p.wait_time for p in sim.have_arrived]
print("Travel:", np.average(travel_times), np.quantile(travel_times, 0.90))
print("Total:", np.average(total_times), np.quantile(total_times, 0.90))
print("Wait:", np.average(wait_times), np.quantile(wait_times, 0.90))
assert 15 < np.average(travel_times) < 24
assert 30 < np.quantile(travel_times, 0.9) < 40
assert 35 < np.average(total_times) < 48
assert 50 < np.quantile(total_times, 0.9) < 75
assert 18 < np.average(wait_times) < 25
assert 30 < np.quantile(wait_times, 0.9) < 48

"""### Two buses

Good.  Let us try to make life easier for our passengers, and increase the number of buses to two.  Will this cut the wait time in half?
"""

sim = Simulation(trace=False, person_rate=2,
                 num_stops=12,
                 bus_nextstop_time=2, bus_max_capacity=50, num_buses=2)

sim.start()

for _ in range(5000):
    sim.step()

sim.plot()

"""Interesting.  What happens to wait and travel times? What do the buses do?  How do you explain the result?  What insight do you derive from the simulation?

Here are some tests.
"""

### 30 points: Tests for two buses

sim1 = Simulation(trace=False, person_rate=2,
            num_stops=12,
            bus_nextstop_time=2, bus_max_capacity=50, num_buses=1)

sim2 = Simulation(trace=False, person_rate=2,
            num_stops=12,
            bus_nextstop_time=2, bus_max_capacity=50, num_buses=2)

sim1.start()
for _ in range(10000):
    sim1.step()

check_simulation(sim2)
