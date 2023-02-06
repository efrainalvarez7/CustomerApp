import javax.swing.*;
import java.util.Scanner;


public class Main {
    public static void main (String[] args){
        //the window frame
        JFrame myFrame = new JFrame("Welcome to Italian Resteruant");
        myFrame.setSize(400,400);
        myFrame.setVisible(true);
        //design of the window
        JPanel panel1 = new JPanel();
        JButton button1 = new JButton("Set Reservation"); 
        panel1.add(button1);
        myFrame.add(panel1);
        myFrame.setVisible(true);
        //button action after click 
        button1.addActionListener(e -> System.out.println("this is not working"));
        myFrame.setVisible(true);
    }
}