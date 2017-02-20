import tournament
import game_agent

NUM_MATCHES = 100 # number of matches against each opponent

def main():

    HEURISTICS = [("Null", tournament.null_score),
                  ("Open", tournament.open_move_score),
                  ("Improved", tournament.improved_score)]
    AB_ARGS = {"search_depth": 5, "method": 'alphabeta', "iterative": False}
    MM_ARGS = {"search_depth": 3, "method": 'minimax', "iterative": False}
    CUSTOM_ARGS = {"method": 'alphabeta', 'iterative': True}

    # Create a collection of CPU agents using fixed-depth minimax or alpha beta
    # search, or random selection.  The agent names encode the search method
    # (MM=minimax, AB=alpha-beta) and the heuristic function (Null=null_score,
    # Open=open_move_score, Improved=improved_score). For example, MM_Open is
    # an agent using minimax search with the open moves heuristic.
    mm_agents = [tournament.Agent(tournament.CustomPlayer(score_fn=h, **MM_ARGS),
                       "MM_" + name) for name, h in HEURISTICS]
    ab_agents = [tournament.Agent(tournament.CustomPlayer(score_fn=h, **AB_ARGS),
                       "AB_" + name) for name, h in HEURISTICS]
    random_agents = [tournament.Agent(tournament.RandomPlayer(), "Random")]

    # ID_Improved agent is used for comparison to the performance of the
    # submitted agent for calibration on the performance across different
    # systems; i.e., the performance of the student agent is considered
    # relative to the performance of the ID_Improved agent to account for
    # faster or slower computers.
    test_agents = []

    #test_agents.append(tournament.Agent(tournament.CustomPlayer(score_fn=tournament.close_to_center2, **CUSTOM_ARGS), "Student Close to center2"))
    #test_agents.append(tournament.Agent(tournament.CustomPlayer(score_fn=tournament.close_to_oponent2, **CUSTOM_ARGS),
    #                                        "Student close_to_oponent2"))
    test_agents.append(tournament.Agent(tournament.CustomPlayer(score_fn=game_agent.close_to_center3, **CUSTOM_ARGS),
                                        "Student CloseToCenter3"))
    #test_agents.append(tournament.Agent(tournament.CustomPlayer(score_fn=game_agent.close_to_center2, **CUSTOM_ARGS),
    #                                    "Student CloseToCenter2"))
    #test_agents.append(tournament.Agent(tournament.CustomPlayer(score_fn=tournament.mixed_new4, **CUSTOM_ARGS),
    #                                    "Student mixed new4"))
    #test_agents.append(tournament.Agent(tournament.CustomPlayer(score_fn=tournament.my_moves_vs_opponent_score3, **CUSTOM_ARGS),
    #                                    "Student my_moves_vs_opponent_score3"))
    #test_agents.append(tournament.Agent(tournament.CustomPlayer(score_fn=tournament.my_moves_vs_opponent_score4, **CUSTOM_ARGS),
    #                                    "Student my_moves_vs_opponent_score4"))
    #test_agents.append(tournament.Agent(tournament.CustomPlayer(score_fn=tournament.my_moves_vs_opponent_score5, **CUSTOM_ARGS),
    #                                    "Student my_moves_vs_opponent_score5"))
    test_agents.append(tournament.Agent(tournament.CustomPlayer(score_fn=game_agent.mixed_new, **CUSTOM_ARGS),
                                        "Student mixed new"))

    #test_agents.append(tournament.Agent(tournament.CustomPlayer(score_fn=tournament.close_to_oponent, **CUSTOM_ARGS),
    #                                    "Student close_to_oponent"))
    #test_agents.append(tournament.Agent(tournament.CustomPlayer(score_fn=tournament.common_moves_heavy, **CUSTOM_ARGS),
    #                                    "Student Common Moves Heavy"))
    #test_agents.append(tournament.Agent(tournament.CustomPlayer(score_fn=tournament.close_to_center, **CUSTOM_ARGS), "Student Close to center"))
    test_agents.append(tournament.Agent(tournament.CustomPlayer(score_fn=tournament.improved_score, **CUSTOM_ARGS), "ID_Improved"))
    #test_agents.append(tournament.Agent(tournament.CustomPlayer(score_fn=tournament.common_moves, **CUSTOM_ARGS), "Student common_moves"))
    #test_agents.append(tournament.Agent(tournament.CustomPlayer(score_fn=tournament.my_moves_vs_opponent_score, **CUSTOM_ARGS), "Student my_moves_vs_opponent_score"))

    iterations = 10

    for agentUT in test_agents:
        print("")
        print("*************************")
        print("{:^25}".format("Evaluating: " + agentUT.name))
        print("*************************")

        agents = random_agents + mm_agents + ab_agents + [agentUT]
        win_ratio = tournament.play_round(agents, NUM_MATCHES)

        print("\n\nResults:")
        print("----------")
        print("{!s:<15}{:>10.2f}%".format(agentUT.name, win_ratio))


if __name__ == "__main__":
    main()
