import numpy as np
import torch


class VRPInstance():
    def __init__(self, nb_customers, locations, original_locations, demand, capacity, use_cost_memory=True):
        self.nb_customers = nb_customers
        self.locations = locations  # coordinates of all locations in the interval [0, 1]
        self.original_locations = original_locations  # original coordinates of locations (used to compute objective
        # value)
        self.demand = demand  # demand for each customer (integer). Values are divided by capacity right before being
        # fed to the network
        self.capacity = capacity  # capacity of the vehicle

        self.solution = None  # List of tours. Each tour is a list of location elements. Each location element is a
        # list with three values [i_l, d, i_n], with i_l being the index of the location (0=depot),
        # d being the fulfilled demand of customer i_l by that tour, and
        # i_n being the index of the associated network input.
        self.nn_input_idx_to_tour = None  # After get_network_input() has been called this is a list where the
        # i-th element corresponds to the tour end represented by the i-th network input. If the network
        # points to an input, this allows us to find out, which tour end that input corresponds to.
        self.open_nn_input_idx = None  # List of idx of those nn_inputs that have not been visited
        self.incomplete_tours = None  # List of incomplete tours of self.solution
        if use_cost_memory:
            self.costs_memory = np.full((nb_customers + 1, nb_customers + 1), np.nan, dtype="float")
        else:
            self.costs_memory = None

    def get_n_closest_locations_to(self, origin_location_id, mask, n):
        """Return the idx of the n closest locations sorted by distance."""
        distances = np.array([np.inf] * len(mask))
        distances[mask] = ((self.locations[mask] * self.locations[origin_location_id]) ** 2).sum(1)
        order = np.argsort(distances)
        return order[:n]

    def create_initial_solution(self):
        """Create an initial solution for this instance using a greedy heuristic."""
        self.solution = [[[0, 0, 0]], [[0, 0, 0]]]
        cur_load = self.capacity
        mask = np.array([True] * (self.nb_customers + 1))
        mask[0] = False
        while mask.any():
            closest_customer_idx = self.get_n_closest_locations_to(self.solution[-1][-1][0], mask, 1)[0]
            if self.demand[closest_customer_idx] <= cur_load:
                mask[closest_customer_idx] = False
                self.solution[-1].append([int(closest_customer_idx), int(self.demand[closest_customer_idx]), None])
                cur_load -= self.demand[closest_customer_idx]
            else:
                self.solution[-1].append([0, 0, 0])
                self.solution.append([[0, 0, 0]])
                cur_load = self.capacity
        self.solution[-1].append([0, 0, 0])

    def get_costs_memory(self, round):
        """Return the cost of the current complete solution. Uses a memory to improve performance."""
        c = 0
        for t in self.solution:
            if t[0][0] != 0 or t[-1][0] != 0:
                raise Exception("Incomplete solution.")
            for i in range(0, len(t) - 1):
                from_idx = t[i][0]
                to_idx = t[i + 1][0]
                if np.isnan(self.costs_memory[from_idx, to_idx]):
                    cc = np.sqrt((self.original_locations[from_idx, 0] - self.original_locations[to_idx, 0]) ** 2
                                 + (self.original_locations[from_idx, 1] - self.original_locations[to_idx, 1]) ** 2)
                    if round:
                        cc = np.round(cc)
                    self.costs_memory[from_idx, to_idx] = cc
                    c += cc
                else:
                    c += self.costs_memory[from_idx, to_idx]
        return c

    def get_costs(self, round):
        """Return the cost of the current complete solution."""
        c = 0
        for t in self.solution:
            if t[0][0] != 0 or t[-1][0] != 0:
                raise Exception("Incomplete solution.")
            for i in range(0, len(t) - 1):
                cc = np.sqrt((self.original_locations[t[i][0], 0] - self.original_locations[t[i + 1][0], 0]) ** 2
                             + (self.original_locations[t[i][0], 1] - self.original_locations[t[i + 1][0], 1]) ** 2)
                if round:
                    cc = np.round(cc)
                c += cc
        return c

    def get_costs_incomplete(self, round):
        """Return the cost of the current incomplete solution."""
        c = 0
        for tour in self.solution:
            if len(tour) <= 1:
                continue
            for i in range(0, len(tour) - 1):
                cc = np.sqrt((self.original_locations[tour[i][0], 0] - self.original_locations[tour[i + 1][0], 0]) ** 2
                             + (self.original_locations[tour[i][0], 1] - self.original_locations[
                    tour[i + 1][0], 1]) ** 2)
                if round:
                    cc = np.round(cc)
                c += cc
        return c

    def destroy(self, customers_to_remove_idx):
        """Remove the customers with the given idx from their tours. This creates an incomplete solution."""
        self.incomplete_tours = []
        st = []  # solution tours

        removed_customer_idx = []

        for tour in self.solution:
            last_split_idx = 0
            for i in range(1, len(tour) - 1):
                if tour[i][0] in customers_to_remove_idx:
                    # Create two new tours:
                    # The first consisting of the tour from the depot or from the last removed customer to the
                    # customer that should be removed
                    if i > last_split_idx and i > 1:
                        new_tour_pre = tour[last_split_idx:i]
                        st.append(new_tour_pre)
                        self.incomplete_tours.append(new_tour_pre)

                    # The second consisting of only the customer to be removed
                    customer_idx = tour[i][0]
                    if customer_idx not in removed_customer_idx:  # make sure the customer has not already been
                        # extracted from a different tour
                        demand = int(self.demand[customer_idx])
                        new_tour = [[customer_idx, demand, None]]
                        st.append(new_tour)
                        self.incomplete_tours.append(new_tour)
                        removed_customer_idx.append(customer_idx)
                    last_split_idx = i + 1

            if last_split_idx > 0:
                # Create another new tour consisting of the remaining part of the original tour
                if last_split_idx < len(tour) - 1:
                    new_tour_post = tour[last_split_idx:]
                    st.append(new_tour_post)
                    self.incomplete_tours.append(new_tour_post)
            else:  # add unchanged tour
                st.append(tour)

        self.solution = st

    def destroy_random(self, p):
        """Random destroy. Select customers that should be removed at random and remove them from tours."""
        customers_to_remove_idx = np.random.choice(range(1, self.nb_customers + 1), int(self.nb_customers * p),
                                                   replace=False)
        self.destroy(customers_to_remove_idx)

    def destroy_point_based(self, p):
        """Point based destroy. Select customers that should be removed based on their distance to a random point
         and remove them from tours."""
        nb_customers_to_remove = int(self.nb_customers * p)
        random_point = np.random.rand(1, 2)
        dist = np.sum((self.locations[1:] - random_point) ** 2, axis=1)
        closest_customers_idx = np.argsort(dist)[:nb_customers_to_remove] + 1
        self.destroy(closest_customers_idx)

    def destroy_tour_based(self, p):
        """Tour based destroy. Remove all tours closest to a randomly selected point from a solution."""

        # Make a dictionary that maps customers to tours
        customer_to_tour = {}
        for i, tour in enumerate(self.solution[1:]):
            for e in tour[1:-1]:
                if e[0] in customer_to_tour:
                    customer_to_tour[e[0]].append(i + 1)
                else:
                    customer_to_tour[e[0]] = [i + 1]

        nb_customers_to_remove = int(self.nb_customers * p)  # Number of customer that should be removed
        nb_removed_customers = 0
        tours_to_remove_idx = []
        random_point = np.random.rand(1, 2)  # Randomly selected point
        dist = np.sum((self.locations[1:] - random_point) ** 2, axis=1)
        closest_customers_idx = np.argsort(dist) + 1

        # Iterate over customers starting with the customer closest to the random point.
        for customer_idx in closest_customers_idx:
            # Iterate over the tours of the customer
            for i in customer_to_tour[customer_idx]:
                # and if the tour is not yet marked for removal
                if i not in tours_to_remove_idx:
                    # mark it for removal
                    tours_to_remove_idx.append(i)
                    nb_removed_customers += len(self.solution[i])

            # Stop once enough tours are marked for removal
            if nb_removed_customers >= nb_customers_to_remove and len(tours_to_remove_idx) > 1:
                break

        # Create the new tours that all consist of only a single customer
        new_tours = []
        removed_customer_idx = []
        for i in tours_to_remove_idx:
            tour = self.solution[i]
            for e in tour[1:-1]:
                if e[0] in removed_customer_idx:
                    for new_tour in new_tours:
                        if new_tour[0][0] == e[0]:
                            new_tour[0][1] += e[1]
                            break
                else:
                    new_tours.append([e])
                    removed_customer_idx.append(e[0])

        # Remove the tours that are marked for removal from the solution
        for index in sorted(tours_to_remove_idx, reverse=True):
            del self.solution[index]

        self.solution.extend(new_tours)  # Add new tours to solution
        self.incomplete_tours = new_tours

    def _get_incomplete_tours(self):
        incomplete_tours = []
        for tour in self.solution:
            if tour[0][0] != 0 or tour[-1][0] != 0:
                incomplete_tours.append(tour)
        return incomplete_tours

    def get_max_nb_input_points(self):
        incomplete_tours = self.incomplete_tours
        nb = 1  # input point for the depot
        for tour in incomplete_tours:
            if len(tour) == 1:
                nb += 1
            else:
                if tour[0][0] != 0:
                    nb += 1
                if tour[-1][0] != 0:
                    nb += 1
        return nb

    def get_network_input(self, input_size):
        """Generate the tensor representation of an incomplete solution (i.e, a representation of the repair problem).
         The input size must be provided so that the representations of all inputs of the batch have the same size.

        [:, 0] x-coordinates for all points
        [:, 1] y-coordinates for all points
        [:, 2] demand values for all points
        [:, 3] state values for all points

        """
        nn_input = np.zeros((input_size, 4))
        nn_input[0, :2] = self.locations[0]  # Depot location
        nn_input[0, 2] = -1 * self.capacity  # Depot demand
        nn_input[0, 3] = -1  # Depot state
        network_input_idx_to_tour = [None] * input_size
        network_input_idx_to_tour[0] = [self.solution[0], 0]
        i = 1
        destroyed_location_idx = []

        incomplete_tours = self.incomplete_tours
        for tour in incomplete_tours:
            # Create an input for a tour consisting of a single customer
            if len(tour) == 1:
                nn_input[i, :2] = self.locations[tour[0][0]]
                nn_input[i, 2] = tour[0][1]
                nn_input[i, 3] = 1
                tour[0][2] = i
                network_input_idx_to_tour[i] = [tour, 0]
                destroyed_location_idx.append(tour[0][0])
                i += 1
            else:
                # Create an input for the first location in an incomplete tour if the location is not the depot
                if tour[0][0] != 0:
                    nn_input[i, :2] = self.locations[tour[0][0]]
                    nn_input[i, 2] = sum(l[1] for l in tour)
                    network_input_idx_to_tour[i] = [tour, 0]
                    if tour[-1][0] == 0:
                        nn_input[i, 3] = 3
                    else:
                        nn_input[i, 3] = 2
                    tour[0][2] = i
                    destroyed_location_idx.append(tour[0][0])
                    i += 1
                # Create an input for the last location in an incomplete tour if the location is not the depot
                if tour[-1][0] != 0:
                    nn_input[i, :2] = self.locations[tour[-1][0]]
                    nn_input[i, 2] = sum(l[1] for l in tour)
                    network_input_idx_to_tour[i] = [tour, len(tour) - 1]
                    tour[-1][2] = i
                    if tour[0][0] == 0:
                        nn_input[i, 3] = 3
                    else:
                        nn_input[i, 3] = 2
                    destroyed_location_idx.append(tour[-1][0])
                    i += 1

        self.open_nn_input_idx = list(range(1, i))
        self.nn_input_idx_to_tour = network_input_idx_to_tour
        return nn_input[:, :2], nn_input[:, 2:]

    def _get_network_input_update_for_tour(self, tour, new_demand):
        """Returns an nn_input update for the tour tour. The demand of the tour is updated to new_demand"""
        nn_input_idx_start = tour[0][2]  # Idx of the nn_input for the first location in tour
        nn_input_idx_end = tour[-1][2]  # Idx of the nn_input for the last location in tour

        # If the tour stars and ends at the depot, no update is required
        if nn_input_idx_start == 0 and nn_input_idx_end == 0:
            return []

        nn_input_update = []
        # Tour with a single location
        if len(tour) == 1:
            if tour[0][0] != 0:
                nn_input_update.append([nn_input_idx_end, new_demand, 1])
                self.nn_input_idx_to_tour[nn_input_idx_end] = [tour, 0]
        else:
            # Tour contains the depot
            if tour[0][0] == 0 or tour[-1][0] == 0:
                # First location in the tour is not the depot
                if tour[0][0] != 0:
                    nn_input_update.append([nn_input_idx_start, new_demand, 3])
                    # update first location
                    self.nn_input_idx_to_tour[nn_input_idx_start] = [tour, 0]
                # Last location in the tour is not the depot
                elif tour[-1][0] != 0:
                    nn_input_update.append([nn_input_idx_end, new_demand, 3])
                    # update last location
                    self.nn_input_idx_to_tour[nn_input_idx_end] = [tour, len(tour) - 1]
            # Tour does not contain the depot
            else:
                # update first and last location of the tour
                nn_input_update.append([nn_input_idx_start, new_demand, 2])
                self.nn_input_idx_to_tour[nn_input_idx_start] = [tour, 0]
                nn_input_update.append([nn_input_idx_end, new_demand, 2])
                self.nn_input_idx_to_tour[nn_input_idx_end] = [tour, len(tour) - 1]
        return nn_input_update

    def do_action(self, id_from, id_to):
        """Performs an action. The tour end represented by input with the id id_from is connected to the tour end
         presented by the input with id id_to."""
        tour_from = self.nn_input_idx_to_tour[id_from][0]  # Tour that should be connected
        tour_to = self.nn_input_idx_to_tour[id_to][0]  # to this tour.
        pos_from = self.nn_input_idx_to_tour[id_from][1]  # Position of the location that should be connected in tour_from
        pos_to = self.nn_input_idx_to_tour[id_to][1]  # Position of the location that should be connected in tour_to

        nn_input_update = []  # Instead of recalculating the tensor representation, we only compute an update description.
        # This improves performance.

        # Exchange tour_from with tour_to or invert order of the tours. This reduces the number of cases that need
        # to be considered in the following.
        if len(tour_from) > 1 and len(tour_to) > 1:
            if pos_from > 0 and pos_to > 0:
                tour_to.reverse()
            elif pos_from == 0 and pos_to == 0:
                tour_from.reverse()
            elif pos_from == 0 and pos_to > 0:
                tour_from, tour_to = tour_to, tour_from
        elif len(tour_to) > 1:
            if pos_to == 0:
                tour_to.reverse()
            tour_from, tour_to = tour_to, tour_from
        elif len(tour_from) > 1 and pos_from == 0:
            tour_from.reverse()

        # Now we only need to consider two cases 1) Connecting an incomplete tour with more than one location
        # to an incomplete tour with more than one location 2) Connecting an incomplete tour (single
        # or multiple locations) to incomplete tour consisting of a single location

        # Case 1
        if len(tour_from) > 1 and len(tour_to) > 1:
            combined_demand = sum(l[1] for l in tour_from) + sum(l[1] for l in tour_to)
            assert combined_demand <= self.capacity  # This is ensured by the masking schema

            # The two incomplete tours are combined to one (in)complete tour. All network inputs associated with the
            # two connected tour ends are set to 0
            nn_input_update.append([tour_from[-1][2], 0, 0])
            nn_input_update.append([tour_to[0][2], 0, 0])
            tour_from.extend(tour_to)
            self.solution.remove(tour_to)
            nn_input_update.extend(self._get_network_input_update_for_tour(tour_from, combined_demand))

        # Case 2
        if len(tour_to) == 1:
            demand_from = sum(l[1] for l in tour_from)
            combined_demand = demand_from + sum(l[1] for l in tour_to)
            unfulfilled_demand = combined_demand - self.capacity

            # The new tour has a total demand that is smaller than or equal to the vehicle capacity
            if unfulfilled_demand <= 0:
                if len(tour_from) > 1:
                    nn_input_update.append([tour_from[-1][2], 0, 0])
                # Update solution
                tour_from.extend(tour_to)
                self.solution.remove(tour_to)
                # Generate input update
                nn_input_update.extend(self._get_network_input_update_for_tour(tour_from, combined_demand))
            # The new tour has a total demand that is larger than the vehicle capacity
            else:
                nn_input_update.append([tour_from[-1][2], 0, 0])
                if len(tour_from) > 1 and tour_from[0][0] != 0:
                    nn_input_update.append([tour_from[0][2], 0, 0])

                # Update solution
                tour_from.append([tour_to[0][0], tour_to[0][1], tour_to[0][2]])  # deepcopy of tour_to
                tour_from[-1][1] = self.capacity - demand_from
                tour_from.append([0, 0, 0])
                if tour_from[0][0] != 0:
                    tour_from.insert(0, [0, 0, 0])
                tour_to[0][1] = unfulfilled_demand  # Update demand of tour_to

                nn_input_update.extend(self._get_network_input_update_for_tour(tour_to, unfulfilled_demand))

        # Add depot tour to the solution tours if it was removed
        if self.solution[0] != [[0, 0, 0]]:
            self.solution.insert(0, [[0, 0, 0]])
            self.nn_input_idx_to_tour[0] = [self.solution[0], 0]

        for update in nn_input_update:
            if update[2] == 0 and update[0] != 0:
                self.open_nn_input_idx.remove(update[0])

        return nn_input_update, tour_from[-1][2]

    def verify_solution(self, config):
        """Verify that a feasible solution has been found."""
        d = np.zeros((self.nb_customers + 1), dtype=int)
        for i in range(len(self.solution)):
            for ii in range(len(self.solution[i])):
                d[self.solution[i][ii][0]] += self.solution[i][ii][1]
        if (self.demand != d).any():
            raise Exception('Solution could not be verified.')

        for tour in self.solution:
            if sum([t[1] for t in tour]) > self.capacity:
                raise Exception('Solution could not be verified.')

        if not config.split_delivery:
            customers = []
            for tour in self.solution:
                for c in tour:
                    if c[0] != 0:
                        customers.append(c[0])

            if len(customers) > len(set(customers)):
                raise Exception('Solution could not be verified.')

    def get_solution_copy(self):
        """ Returns a copy of self.solution"""
        solution_copy = []
        for tour in self.solution:
            solution_copy.append([x[:] for x in tour]) # Fastest way to make a deep copy
        return solution_copy

    def __deepcopy__(self, memo):
        solution_copy = self.get_solution_copy()
        new_instance = VRPInstance(self.nb_customers, self.locations, self.original_locations, self.demand,
                                   self.capacity)
        new_instance.solution = solution_copy
        new_instance.costs_memory = self.costs_memory

        return new_instance


def get_mask(origin_nn_input_idx, dynamic_input, instances, config, capacity):
    """ Returns a mask for the current nn_input"""
    batch_size = origin_nn_input_idx.shape[0]

    # Start with all used input positions
    mask = (dynamic_input[:, :, 1] != 0).cpu().long().numpy()

    for i in range(batch_size):
        idx_from = origin_nn_input_idx[i]
        origin_tour = instances[i].nn_input_idx_to_tour[idx_from][0]
        origin_pos = instances[i].nn_input_idx_to_tour[idx_from][1]

        # Find the start of the tour in the nn input
        # e.g. for the tour [2, 3] two entries in nn input exists
        if origin_pos == 0:
            idx_same_tour = origin_tour[-1][2]
        else:
            idx_same_tour = origin_tour[0][2]

        mask[i, idx_same_tour] = 0

        # Do not allow origin location = destination location
        mask[i, idx_from] = 0

    mask = torch.from_numpy(mask)

    origin_tour_demands = dynamic_input[torch.arange(batch_size), origin_nn_input_idx, 0]
    combined_demand = origin_tour_demands.unsqueeze(1).expand(batch_size, dynamic_input.shape[1]) + dynamic_input[:, :,
                                                                                                    0]

    if config.split_delivery:
        multiple_customer_tour = (dynamic_input[torch.arange(batch_size), origin_nn_input_idx, 1] > 1).unsqueeze(1).expand(
            batch_size, dynamic_input.shape[1])

        # If the origin tour consists of multiple customers mask all tours with multiple customers where
        # the combined demand is > 1
        mask[multiple_customer_tour & (combined_demand > capacity) & (dynamic_input[:, :, 1] > 1)] = 0

        # If the origin tour consists of a single customer mask all tours with demand is >= 1
        mask[(~multiple_customer_tour) & (dynamic_input[:, :, 0] >= capacity)] = 0
    else:
        mask[combined_demand > capacity] = 0

    mask[:, 0] = 1  # Always allow to go to the depot

    return mask
